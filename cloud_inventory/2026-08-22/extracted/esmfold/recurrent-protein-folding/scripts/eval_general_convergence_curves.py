#!/usr/bin/env python3
"""Evaluate general belief probes as convergence curves on cached proteins.

This runs forward passes only. For each protein, it samples residue pairs,
computes expected distances from the general probe at each layer, compares them
to the final trained general readout (layer 46), and writes the same layerwise
curve format used by the plotting scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


def load_probe(path: Path, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def sample_pairs(length: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed + length)
    i = rng.integers(0, length, size=count)
    j = rng.integers(0, length - 1, size=count)
    j = j + (j >= i)
    return torch.from_numpy(i).long(), torch.from_numpy(j).long()


def expected_from_probe(
    probe: PairDistogramBottleneck,
    z: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    centers: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    length = z.shape[0]
    with torch.no_grad():
        logits = probe.encode(z[i, j].to(device, torch.float32), i.to(device), j.to(device), length)
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.detach().cpu().numpy()


def expected_from_native_head(
    head: torch.nn.Module,
    z: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    centers: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        pair = z[i, j].to(device, torch.float32)
        pair_t = z[j, i].to(device, torch.float32)
        logits = head(0.5 * (pair + pair_t))
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.detach().cpu().numpy()


def find_distogram_head(model: EsmForProteinFolding) -> torch.nn.Module:
    for attr in ("distogram_head", "disto_head"):
        if hasattr(model.trunk, attr):
            return getattr(model.trunk, attr)
    for name, mod in model.named_modules():
        if "distogram" in name.lower() and isinstance(mod, nn.Module) and any(
            isinstance(c, nn.Linear) for c in mod.modules()
        ):
            print(f"using ESMFold native distogram head at {name}", flush=True)
            return mod
    raise RuntimeError("Could not locate ESMFold native distogram head")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def clean_name(path: Path) -> str:
    name = path.name
    for prefix in ("benchmark_", "hard_synthetic_miss_", "synthetic_design_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def metrics_for_run_dir(run_dir: Path) -> dict:
    p = run_dir / "analysis/esmfold_direct_eval/metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general-probes-root", type=Path, required=True)
    ap.add_argument("--protein-dirs-file", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=32768)
    ap.add_argument("--max-length", type=int, default=10000)
    ap.add_argument("--reference", choices=["final-trained-distogram", "native-output"], default="final-trained-distogram")
    ap.add_argument("--start-layer", type=int, default=None)
    ap.add_argument("--end-layer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.out_root.mkdir(parents=True, exist_ok=True)
    centers = distogram_bin_centers(device, torch.float32)
    native_head = None
    if args.reference == "native-output":
        print("loading ESMFold native distogram head...", flush=True)
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        native_head = find_distogram_head(model).to(device).eval()
        for p in native_head.parameters():
            p.requires_grad_(False)
        del model
        torch.cuda.empty_cache()

    protein_dirs = [Path(x.strip()) for x in args.protein_dirs_file.read_text().splitlines() if x.strip()]
    available_layers = sorted(
        int(p.parent.name.split("_")[-1])
        for p in args.general_probes_root.glob("layer_*/probe.pt")
    )
    if not available_layers:
        raise SystemExit(f"No probe.pt files found under {args.general_probes_root}")
    start_layer = args.start_layer if args.start_layer is not None else min(available_layers)
    end_layer = args.end_layer if args.end_layer is not None else max(available_layers)
    layers_to_eval = [layer for layer in available_layers if start_layer <= layer <= end_layer]
    if not layers_to_eval:
        raise SystemExit(f"No probes available in requested layer range {start_layer}..{end_layer}")
    print(f"evaluating probe layers {layers_to_eval[0]}..{layers_to_eval[-1]} ({len(layers_to_eval)} layers)", flush=True)
    summary = []
    for idx, run_dir in enumerate(protein_dirs, 1):
        cache_p = run_dir / "esmfold_activations.pt"
        if not cache_p.exists():
            continue
        meta = torch.load(cache_p, map_location="cpu", weights_only=False, mmap=True)
        if "length" not in meta:
            print(f"[{idx:02d}/{len(protein_dirs)}] skip {clean_name(run_dir)} no length metadata", flush=True)
            del meta
            continue
        length_meta = int(meta["length"])
        del meta
        if length_meta > args.max_length:
            print(f"[{idx:02d}/{len(protein_dirs)}] skip {clean_name(run_dir)} L={length_meta} > max_length={args.max_length}", flush=True)
            continue
        cache = torch.load(cache_p, map_location="cpu", weights_only=False, mmap=True)
        z_full = cache["z"].squeeze(1)
        length = int(z_full.shape[1])
        name = clean_name(run_dir)
        group = "natural" if run_dir.name.startswith("benchmark_") else "synthetic"
        i, j = sample_pairs(length, args.pairs, args.seed)
        print(f"[{idx:02d}/{len(protein_dirs)}] {name} {group} L={length} pairs={len(i)}", flush=True)

        if args.reference == "native-output":
            assert native_head is not None
            final_ed = expected_from_native_head(native_head, z_full[-1], i, j, centers, device)
            reference_label = "native_output"
        else:
            final_probe = load_probe(args.general_probes_root / "layer_46" / "probe.pt", device)
            final_ed = expected_from_probe(final_probe, z_full[46], i, j, centers, device)
            del final_probe
            torch.cuda.empty_cache()
            reference_label = "final_trained_distogram"
        layer_eds: list[np.ndarray] = []
        rows = []
        for layer in layers_to_eval:
            probe = load_probe(args.general_probes_root / f"layer_{layer:02d}" / "probe.pt", device)
            ed = expected_from_probe(probe, z_full[layer], i, j, centers, device)
            del probe
            torch.cuda.empty_cache()
            layer_eds.append(ed)
            c_final = corr(ed, final_ed)
            mae_final = float(np.abs(ed - final_ed).mean())
            rows.append(
                {
                    "layer": layer,
                    "reference": reference_label,
                    "corr_vs_final_distogram": c_final,
                    "dist_mae_vs_final_distogram": mae_final,
                    # Backward-compatible names for older plotters. In this
                    # general-probe evaluator, "47" means the final trained
                    # distogram readout, not ESMFold's native model output.
                    "corr_vs_47": c_final,
                    "dist_mae_vs_47": mae_final,
                    "dist_mae_vs_centroid": "",
                    "corr_vs_centroid": "",
                }
            )
        if args.reference == "native-output":
            layer_eds.append(final_ed)
            rows.append(
                {
                    "layer": 47,
                    "reference": reference_label,
                    "corr_vs_final_distogram": 1.0,
                    "dist_mae_vs_final_distogram": 0.0,
                    "corr_vs_47": 1.0,
                    "dist_mae_vs_47": 0.0,
                    "dist_mae_vs_centroid": "",
                    "corr_vs_centroid": "",
                }
            )

        all_ed = np.stack(layer_eds)
        centroid = all_ed.mean(axis=0)
        for row, ed in zip(rows, all_ed):
            row["dist_mae_vs_centroid"] = float(np.abs(ed - centroid).mean())
            row["corr_vs_centroid"] = corr(ed, centroid)

        out_dir = args.out_root / f"{name}_analysis"
        (out_dir / "esmfold_direct_eval").mkdir(parents=True, exist_ok=True)
        with (out_dir / "layerwise_distance_correlations.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        metrics = metrics_for_run_dir(run_dir)
        if metrics:
            (out_dir / "esmfold_direct_eval/metrics.json").write_text(json.dumps(metrics, indent=2))

        corr_arr = np.array([r["corr_vs_final_distogram"] for r in rows])
        layers_arr = np.array([int(r["layer"]) for r in rows])
        measured = layers_arr <= 46
        summary.append(
            {
                "group": group,
                "protein": name,
                "length": length,
                "mae": metrics.get("distance_mae_all_pairs", ""),
                "tm_score": "",
                "plddt": metrics.get("mean_plddt", ""),
                "min_last5": float(np.nanmin(corr_arr[(layers_arr >= max(layers_to_eval) - 4) & measured])),
                "late_amp": float(
                    np.nanmax(corr_arr[(layers_arr >= max(layers_to_eval) - 11) & measured])
                    - np.nanmin(corr_arr[(layers_arr >= max(layers_to_eval) - 11) & measured])
                ),
            }
        )
        del z_full
        torch.cuda.empty_cache()

    with (args.out_root / "general_convergence_summary.csv").open("w", newline="") as f:
        fields = ["group", "protein", "length", "mae", "tm_score", "plddt", "min_last5", "late_amp"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
