#!/usr/bin/env python3
"""Cache frozen ESMFold trunk activations for every protein in a TSV manifest."""

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
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture


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
    pred_dist = pairwise(pred_ca)
    gt_dist = pairwise(gt_ca)
    iu = np.triu_indices(n, k=1)
    dp = pred_dist[iu]
    dt = gt_dist[iu]
    aligned = kabsch_align(pred_ca, gt_ca)
    ca_err = np.sqrt(((aligned - gt_ca) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    tm_like = float(np.mean(1.0 / (1.0 + (ca_err / d0) ** 2)))
    corr = float(np.corrcoef(dp, dt)[0, 1]) if len(dp) > 2 and np.nanstd(dp) > 0 and np.nanstd(dt) > 0 else math.nan
    metrics = {
        "length": int(n),
        "ca_rmsd_after_kabsch": float(np.sqrt((ca_err * ca_err).mean())),
        "distance_mae_all_pairs": float(np.abs(dp - dt).mean()),
        "distance_corr_all_pairs": corr,
        "mean_plddt": float(np.nanmean(plddt[:n])),
        "plddt_min": float(np.nanmin(plddt[:n])),
        "plddt_med": float(np.nanmedian(plddt[:n])),
        "plddt_max": float(np.nanmax(plddt[:n])),
        "tm_like_ca": tm_like,
    }
    return metrics, pred_ca


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--train-dirs-file", type=Path, required=True)
    ap.add_argument("--max-proteins", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fold-chunk-size", type=int, default=64)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.train_dirs_file.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))[: args.max_proteins]
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(args.fold_chunk_size)

    train_dirs: list[Path] = []
    for idx, row in enumerate(rows, 1):
        name = row["name"]
        run_dir = args.out_root / name
        cache_p = run_dir / "esmfold_activations.pt"
        gt_p = run_dir / "gt_ca_distances.npz"
        seq_p = run_dir / "sequence.txt"
        metrics_p = run_dir / "analysis/esmfold_direct_eval/metrics.json"
        pred_p = run_dir / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        run_dir.mkdir(parents=True, exist_ok=True)
        train_dirs.append(run_dir)
        if cache_p.exists() and seq_p.exists() and metrics_p.exists():
            print(f"[{idx:02d}/{len(rows)}] {name} cache exists", flush=True)
            continue
        seq, ca, chain = extract_sequence_and_ca(Path(row["gt_pdb"]), row.get("chain", ""))
        np.savez(gt_p, distances=pairwise(ca), ca_coords=ca, sequence=seq)
        seq_p.write_text(seq)
        (run_dir / "source.json").write_text(json.dumps({**row, "chain_used": chain, "resolved_length": len(seq)}, indent=2))
        ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        capture = ESMFoldTrunkCapture(model)
        t0 = time.time()
        with capture.capturing(), torch.no_grad():
            out = model(ids)
        metrics, pred_ca = direct_metrics(out, ca)
        cache = {
            "sequence": seq,
            "length": len(seq),
            "z": capture.stacked_z(),
            "seconds": time.time() - t0,
        }
        torch.save(cache, cache_p)
        metrics_dir = metrics_p.parent
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_p.write_text(json.dumps(metrics, indent=2))
        np.savez(pred_p, pred_ca=pred_ca)
        print(
            f"[{idx:02d}/{len(rows)}] {name} L={len(seq)} cached in {cache['seconds']:.1f}s "
            f"MAE={metrics['distance_mae_all_pairs']:.2f} TM={metrics['tm_like_ca']:.3f} pLDDT={metrics['mean_plddt']:.3f}",
            flush=True,
        )

    args.train_dirs_file.write_text("\n".join(str(p) for p in train_dirs) + "\n")
    print(f"wrote {args.train_dirs_file}", flush=True)


if __name__ == "__main__":
    main()
