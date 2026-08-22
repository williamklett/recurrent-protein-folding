#!/usr/bin/env python3
"""Compare general belief probes against per-protein probe readouts.

For each protein and each layer:
  - general readout: shared replay-trained probe_t(z_t), with layer 47 from
    ESMFold's native distogram head
  - local readout: per-protein probe_t(z_t)

The comparison is readout-to-readout, not truth-supervised. It reports
layerwise correlation/MAE between expected-distance matrices, plus whether the
general and local convergence-to-final curves have similar shapes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


def expected_distance(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (F.softmax(logits, dim=-1) * centers).sum(-1)


def apply_probe(probe: PairDistogramBottleneck, z_layer: torch.Tensor, device: torch.device) -> torch.Tensor:
    l = z_layer.shape[0]
    flat = z_layer.reshape(-1, z_layer.shape[-1]).to(device, torch.float32)
    ii, jj = torch.meshgrid(torch.arange(l, device=device), torch.arange(l, device=device), indexing="ij")
    with torch.no_grad():
        logits = probe.encode(flat, ii.flatten(), jj.flatten(), l)
    return logits.reshape(l, l, -1)


def apply_head(head: torch.nn.Module, z_47: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        z = z_47.to(device, torch.float32)
        z = 0.5 * (z + z.transpose(0, 1))
        return head(z)


def load_probe(path: Path, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for param in probe.parameters():
        param.requires_grad_(False)
    return probe


def find_distogram_head(model: torch.nn.Module) -> torch.nn.Module:
    for parent_name in ("trunk", ""):
        parent = getattr(model, parent_name) if parent_name else model
        for attr in ("distogram_head", "disto_head"):
            if hasattr(parent, attr):
                return getattr(parent, attr)
    for name, mod in model.named_modules():
        if "distogram" in name.lower() and isinstance(mod, torch.nn.Module):
            if any(isinstance(child, torch.nn.Linear) for child in mod.modules()):
                return mod
    raise RuntimeError("Could not locate ESMFold distogram head")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general-probes-root", type=Path, required=True)
    ap.add_argument("--protein-dirs-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading ESMFold native distogram head...", flush=True)
    esm = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    head = find_distogram_head(esm).to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    del esm
    torch.cuda.empty_cache()

    print("loading general probes...", flush=True)
    general = {layer: load_probe(args.general_probes_root / f"layer_{layer:02d}" / "probe.pt", device) for layer in range(1, 47)}

    protein_dirs = [Path(line.strip()) for line in args.protein_dirs_file.read_text().splitlines() if line.strip()]
    layer_rows: list[dict] = []
    summary_rows: list[dict] = []

    for idx, d in enumerate(protein_dirs, 1):
        cache_p = d / "esmfold_activations.pt"
        if not cache_p.exists():
            print(f"[{idx:02d}/{len(protein_dirs)}] skip {d}: no cache", flush=True)
            continue
        if not (d / "layer_47/probe.pt").exists():
            print(f"[{idx:02d}/{len(protein_dirs)}] skip {d}: no local probes", flush=True)
            continue

        cache = torch.load(cache_p, map_location="cpu", weights_only=False)
        z_full = cache["z"].squeeze(1).float()
        l = int(z_full.shape[1])
        name = clean_name(d)
        group = "natural" if d.name.startswith("benchmark_") else "synthetic"
        iu = np.triu_indices(l, k=1)
        print(f"[{idx:02d}/{len(protein_dirs)}] {name} group={group} L={l}", flush=True)

        gen_curve: list[float] = []
        loc_curve: list[float] = []
        cross_corrs: list[float] = []
        cross_maes: list[float] = []

        gen_47_logits = apply_head(head, z_full[47], device)
        gen_47 = expected_distance(gen_47_logits, centers).detach().cpu().numpy()[iu]
        local47 = load_probe(d / "layer_47/probe.pt", device)
        loc_47_logits = apply_probe(local47, z_full[47], device)
        loc_47 = expected_distance(loc_47_logits, centers).detach().cpu().numpy()[iu]
        del local47, gen_47_logits, loc_47_logits
        torch.cuda.empty_cache()

        for layer in range(1, 48):
            if layer == 47:
                gen_ed = gen_47
                loc_ed = loc_47
            else:
                local_probe = load_probe(d / f"layer_{layer:02d}" / "probe.pt", device)
                z_layer = z_full[layer]
                gen_logits = apply_probe(general[layer], z_layer, device)
                loc_logits = apply_probe(local_probe, z_layer, device)
                gen_ed = expected_distance(gen_logits, centers).detach().cpu().numpy()[iu]
                loc_ed = expected_distance(loc_logits, centers).detach().cpu().numpy()[iu]
                del local_probe, gen_logits, loc_logits
                torch.cuda.empty_cache()

            cross = corr(gen_ed, loc_ed)
            mae = float(np.abs(gen_ed - loc_ed).mean())
            gen_vs_final = corr(gen_ed, gen_47)
            loc_vs_final = corr(loc_ed, loc_47)
            layer_rows.append(
                {
                    "group": group,
                    "protein": name,
                    "layer": layer,
                    "general_vs_local_corr": cross,
                    "general_vs_local_mae": mae,
                    "general_corr_vs_final": gen_vs_final,
                    "local_corr_vs_final": loc_vs_final,
                    "corr_curve_abs_delta": abs(gen_vs_final - loc_vs_final),
                }
            )
            cross_corrs.append(cross)
            cross_maes.append(mae)
            gen_curve.append(gen_vs_final)
            loc_curve.append(loc_vs_final)

        summary_rows.append(
            {
                "group": group,
                "protein": name,
                "n_layers": len(gen_curve),
                "mean_layer_corr": float(np.nanmean(cross_corrs)),
                "median_layer_corr": float(np.nanmedian(cross_corrs)),
                "min_layer_corr": float(np.nanmin(cross_corrs)),
                "mean_layer_mae": float(np.nanmean(cross_maes)),
                "median_layer_mae": float(np.nanmedian(cross_maes)),
                "curve_corr": corr(np.asarray(gen_curve), np.asarray(loc_curve)),
                "mean_abs_curve_delta": float(np.nanmean(np.abs(np.asarray(gen_curve) - np.asarray(loc_curve)))),
                "general_min_last5": float(np.nanmin(np.asarray(gen_curve)[42:47])),
                "local_min_last5": float(np.nanmin(np.asarray(loc_curve)[42:47])),
                "general_late_amp": float(np.nanmax(np.asarray(gen_curve)[35:47]) - np.nanmin(np.asarray(gen_curve)[35:47])),
                "local_late_amp": float(np.nanmax(np.asarray(loc_curve)[35:47]) - np.nanmin(np.asarray(loc_curve)[35:47])),
            }
        )
        del z_full
        torch.cuda.empty_cache()

    layer_csv = args.out_dir / "layerwise_general_vs_local.csv"
    summary_csv = args.out_dir / "summary_general_vs_local.csv"
    with layer_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        w.writerows(layer_rows)
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {layer_csv}", flush=True)
    print(f"wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
