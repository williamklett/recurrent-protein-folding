#!/usr/bin/env python3
"""Stream general-probe convergence evaluation over a TSV manifest.

Unlike cache_esmfold_activations_for_manifest.py + eval_general_convergence_curves.py,
this does not persist trunk activations. That matters for large batches because
pair activations are hundreds of MB per medium protein.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture, PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def extract_sequence_and_ca(pdb_path: Path, chain_hint: str) -> tuple[str, np.ndarray, str]:
    counts: Counter[str] = Counter()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                counts[ln[21]] += 1
    if not counts:
        raise ValueError(f"no CA atoms in {pdb_path}")
    hint = chain_hint.strip() or " "
    chain = hint if hint in counts and counts[hint] >= 20 else counts.most_common(1)[0][0]
    seq, coords, seen = [], [], set()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
                rs = (int(ln[22:26]), ln[26])
                if rs in seen:
                    continue
                seen.add(rs)
                seq.append(AA3.get(ln[17:20].strip(), "X"))
                coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return "".join(seq), np.asarray(coords, dtype=np.float32), chain


def pairwise(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(axis=0)
    tc = target - target.mean(axis=0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, det]) @ u.T
    return pc @ rot.T + target.mean(axis=0)


def direct_metrics(out: dict, gt_ca: np.ndarray) -> tuple[dict, np.ndarray]:
    plddt = out["plddt"][0, :, 1].detach().cpu().float().numpy()
    if float(np.nanmean(plddt)) > 1.5:
        plddt = plddt / 100.0
    pred_ca = out["positions"][-1, 0, :, 1, :].detach().cpu().float().numpy()
    n = min(len(pred_ca), len(gt_ca))
    pred_ca = pred_ca[:n]
    gt_ca = gt_ca[:n]
    dp = pairwise(pred_ca)
    dt = pairwise(gt_ca)
    iu = np.triu_indices(n, k=1)
    dpv = dp[iu]
    dtv = dt[iu]
    aligned = kabsch_align(pred_ca, gt_ca)
    ca_err = np.sqrt(((aligned - gt_ca) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    metrics = {
        "length": int(n),
        "ca_rmsd_after_kabsch": float(np.sqrt((ca_err * ca_err).mean())),
        "distance_mae_all_pairs": float(np.abs(dpv - dtv).mean()),
        "distance_corr_all_pairs": float(np.corrcoef(dpv, dtv)[0, 1]) if len(dpv) > 2 and np.nanstd(dpv) > 0 and np.nanstd(dtv) > 0 else math.nan,
        "mean_plddt": float(np.nanmean(plddt[:n])),
        "plddt_min": float(np.nanmin(plddt[:n])),
        "plddt_med": float(np.nanmedian(plddt[:n])),
        "plddt_max": float(np.nanmax(plddt[:n])),
        "tm_like_ca": float(np.mean(1.0 / (1.0 + (ca_err / d0) ** 2))),
    }
    return metrics, pred_ca


def find_distogram_head(model: EsmForProteinFolding) -> torch.nn.Module:
    for attr in ("distogram_head", "disto_head"):
        if hasattr(model.trunk, attr):
            return getattr(model.trunk, attr)
    for name, mod in model.named_modules():
        if "distogram" in name.lower() and isinstance(mod, nn.Module) and any(isinstance(c, nn.Linear) for c in mod.modules()):
            print(f"using ESMFold native distogram head at {name}", flush=True)
            return mod
    raise RuntimeError("Could not locate ESMFold native distogram head")


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


def expected_probe(probe: PairDistogramBottleneck, z: torch.Tensor, i: torch.Tensor, j: torch.Tensor, centers: torch.Tensor, device: torch.device) -> np.ndarray:
    length = z.shape[0]
    with torch.no_grad():
        logits = probe.encode(z[i, j].to(device, torch.float32), i.to(device), j.to(device), length)
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.detach().cpu().numpy()


def expected_native(head: torch.nn.Module, z: torch.Tensor, i: torch.Tensor, j: torch.Tensor, centers: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        pair = z[i, j].to(device, torch.float32)
        pair_t = z[j, i].to(device, torch.float32)
        logits = head(0.5 * (pair + pair_t))
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.detach().cpu().numpy()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--general-probes-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--max-proteins", type=int, default=500)
    ap.add_argument("--max-length", type=int, default=700)
    ap.add_argument("--pairs", type=int, default=32768)
    ap.add_argument("--fold-chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))[: args.max_proteins]

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(args.fold_chunk_size)
    native_head = find_distogram_head(model).to(device).eval()
    for p in native_head.parameters():
        p.requires_grad_(False)

    layers = sorted(int(p.parent.name.split("_")[-1]) for p in args.general_probes_root.glob("layer_*/probe.pt"))
    layers = [x for x in layers if 1 <= x <= 46]
    if len(layers) != 46:
        print(f"[warn] expected 46 probes, found {len(layers)} layers {layers[:3]}..{layers[-3:]}", flush=True)
    probes = {layer: load_probe(args.general_probes_root / f"layer_{layer:02d}" / "probe.pt", device) for layer in layers}
    centers = distogram_bin_centers(device, torch.float32)

    summary = []
    for idx, row in enumerate(rows, 1):
        name = row["name"]
        out_dir = args.out_root / f"{name}_analysis"
        corr_p = out_dir / "layerwise_distance_correlations.csv"
        metrics_p = out_dir / "esmfold_direct_eval/metrics.json"
        if corr_p.exists() and metrics_p.exists():
            print(f"[{idx:03d}/{len(rows)}] {name} exists", flush=True)
            metrics = json.loads(metrics_p.read_text())
            summary.append(
                {
                    "group": row.get("class", ""),
                    "protein": name,
                    "length": metrics.get("length", ""),
                    "mae": metrics.get("distance_mae_all_pairs", ""),
                    "tm_score": metrics.get("tm_like_ca", ""),
                    "plddt": metrics.get("mean_plddt", ""),
                    "min_last5": "",
                    "late_amp": "",
                }
            )
            continue
        try:
            seq, gt_ca, chain = extract_sequence_and_ca(Path(row["gt_pdb"]), row.get("chain", ""))
            if len(seq) > args.max_length:
                print(f"[{idx:03d}/{len(rows)}] skip {name} L={len(seq)} > {args.max_length}", flush=True)
                continue
            ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            capture = ESMFoldTrunkCapture(model)
            t0 = time.time()
            with capture.capturing(), torch.no_grad():
                out = model(ids)
            z_full = capture.stacked_z().squeeze(1)
            metrics, pred_ca = direct_metrics(out, gt_ca)
            i, j = sample_pairs(int(z_full.shape[1]), args.pairs, args.seed)
            final_ed = expected_native(native_head, z_full[-1], i, j, centers, device)

            layer_eds = []
            out_rows = []
            for layer in layers:
                ed = expected_probe(probes[layer], z_full[layer], i, j, centers, device)
                layer_eds.append(ed)
                out_rows.append(
                    {
                        "layer": layer,
                        "reference": "native_output",
                        "corr_vs_final_distogram": corr(ed, final_ed),
                        "dist_mae_vs_final_distogram": float(np.abs(ed - final_ed).mean()),
                        "corr_vs_47": corr(ed, final_ed),
                        "dist_mae_vs_47": float(np.abs(ed - final_ed).mean()),
                        "dist_mae_vs_centroid": "",
                        "corr_vs_centroid": "",
                    }
                )
            layer_eds.append(final_ed)
            out_rows.append(
                {
                    "layer": 47,
                    "reference": "native_output",
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
            for out_row, ed in zip(out_rows, all_ed):
                out_row["dist_mae_vs_centroid"] = float(np.abs(ed - centroid).mean())
                out_row["corr_vs_centroid"] = corr(ed, centroid)

            (out_dir / "esmfold_direct_eval").mkdir(parents=True, exist_ok=True)
            (out_dir / "run_artifacts").mkdir(parents=True, exist_ok=True)
            with corr_p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
                w.writeheader()
                w.writerows(out_rows)
            metrics_p.write_text(json.dumps(metrics, indent=2))
            np.savez(out_dir / "esmfold_direct_eval/esmfold_prediction_ca.npz", pred_ca=pred_ca)
            np.savez(out_dir / "run_artifacts/gt_ca_distances.npz", distances=pairwise(gt_ca), ca_coords=gt_ca, sequence=seq)
            (out_dir / "source.json").write_text(json.dumps({**row, "chain_used": chain}, indent=2))

            corr_arr = np.array([float(r["corr_vs_final_distogram"]) for r in out_rows])
            layers_arr = np.array([int(r["layer"]) for r in out_rows])
            measured = layers_arr <= 46
            srow = {
                "group": row.get("class", ""),
                "protein": name,
                "length": metrics["length"],
                "mae": metrics["distance_mae_all_pairs"],
                "tm_score": metrics["tm_like_ca"],
                "plddt": metrics["mean_plddt"],
                "min_last5": float(np.nanmin(corr_arr[(layers_arr >= 42) & measured])),
                "late_amp": float(np.nanmax(corr_arr[(layers_arr >= 35) & measured]) - np.nanmin(corr_arr[(layers_arr >= 35) & measured])),
            }
            summary.append(srow)
            print(
                f"[{idx:03d}/{len(rows)}] {name} L={metrics['length']} "
                f"MAE={metrics['distance_mae_all_pairs']:.2f} TM={metrics['tm_like_ca']:.3f} "
                f"pLDDT={metrics['mean_plddt']:.3f} min5={srow['min_last5']:.3f} "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            del out, z_full, all_ed, layer_eds, final_ed
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[{idx:03d}/{len(rows)}] {name} failed: {exc}", flush=True)

    with (args.out_root / "general_convergence_summary.csv").open("w", newline="") as f:
        fields = ["group", "protein", "length", "mae", "tm_score", "plddt", "min_last5", "late_amp"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
