#!/usr/bin/env python3
"""Run a cheap ESMFold-direct screen over a manifest and emit hard targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding


AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def extract_pdb_chain(pdb_path: Path, chain_hint: str) -> tuple[str, np.ndarray, str]:
    chain_counts: Counter[str] = Counter()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                chain_counts[ln[21]] += 1
    if not chain_counts:
        raise ValueError(f"no CA atoms in {pdb_path}")

    hint = chain_hint.strip() or " "
    chain = hint if hint in chain_counts and chain_counts[hint] >= 20 else chain_counts.most_common(1)[0][0]

    seq_chars: list[str] = []
    coords: list[list[float]] = []
    seen = set()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
                rs = (int(ln[22:26]), ln[26])
                if rs in seen:
                    continue
                seen.add(rs)
                seq_chars.append(AA3.get(ln[17:20].strip(), "X"))
                coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    coords_np = np.asarray(coords, dtype=np.float32)
    return "".join(seq_chars), coords_np, chain


def pairwise(c: np.ndarray) -> np.ndarray:
    d = c[:, None, :] - c[None, :, :]
    return np.sqrt((d * d).sum(-1))


def kabsch_rmsd(pred: np.ndarray, target: np.ndarray) -> float:
    pc = pred - pred.mean(0)
    qc = target - target.mean(0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    aligned = pc @ (vt.T @ np.diag([1, 1, det]) @ u.T).T + target.mean(0)
    return float(np.sqrt(((aligned - target) ** 2).sum(-1).mean()))


def eval_one(model, tokenizer, seq: str, gt_ca: np.ndarray, device: str) -> dict:
    ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids)
    plddt = out["plddt"][0, :, 1].detach().cpu().float()
    mean_plddt = float(plddt.mean())
    if mean_plddt > 1.5:
        mean_plddt /= 100.0
        plddt = plddt / 100.0
    pred_ca = out["positions"][-1, 0, :, 1, :].detach().cpu().numpy()
    n = min(len(pred_ca), len(gt_ca))
    pred_ca = pred_ca[:n]
    gt_ca = gt_ca[:n]
    iu = np.triu_indices(n, k=1)
    dp = pairwise(pred_ca)[iu]
    dt = pairwise(gt_ca)[iu]
    corr = float(np.corrcoef(dp, dt)[0, 1]) if len(dp) > 2 else math.nan
    return {
        "length_eval": n,
        "ca_rmsd_after_kabsch": kabsch_rmsd(pred_ca, gt_ca),
        "distance_mae_all_pairs": float(np.abs(dp - dt).mean()),
        "distance_corr_all_pairs": corr,
        "mean_plddt": mean_plddt,
        "plddt_min": float(plddt.min()),
        "plddt_med": float(plddt.median()),
        "plddt_max": float(plddt.max()),
    }


def read_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ["name", "gt_pdb", "chain", "class", "pdb_id", "length", "release_date", "method", "title", "source", "evidence"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/hard_synthetic_prescreen"))
    ap.add_argument("--selected-manifest", type=Path, default=Path("data/hard_synthetic_misses_manifest.tsv"))
    ap.add_argument("--max-candidates", type=int, default=80)
    ap.add_argument("--select-n", type=int, default=20)
    ap.add_argument("--min-mae", type=float, default=2.0)
    ap.add_argument("--min-rmsd", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.selected_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)[: args.max_candidates]

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(args.device).eval()
    model.esm = model.esm.half()

    out_rows = []
    for i, row in enumerate(rows, 1):
        name = row["name"]
        result_path = args.out_dir / f"{name}.json"
        if result_path.exists():
            metrics = json.loads(result_path.read_text())
        else:
            try:
                seq, gt_ca, chain = extract_pdb_chain(Path(row["gt_pdb"]), row.get("chain", ""))
                metrics = eval_one(model, tokenizer, seq, gt_ca, args.device)
                metrics.update({"name": name, "pdb_id": row.get("pdb_id", name).upper(), "chain_used": chain, "sequence": seq})
                result_path.write_text(json.dumps(metrics, indent=2))
            except Exception as exc:
                print(f"[{i:02d}/{len(rows)}] {name} failed: {exc}", file=sys.stderr)
                continue
        merged = dict(row)
        merged.update(metrics)
        out_rows.append(merged)
        print(
            f"[{i:02d}/{len(rows)}] {name:>5} L={metrics['length_eval']:>4} "
            f"MAE={metrics['distance_mae_all_pairs']:.2f} "
            f"RMSD={metrics['ca_rmsd_after_kabsch']:.2f} "
            f"pLDDT={metrics['mean_plddt']:.3f}"
        )

    out_rows.sort(
        key=lambda r: (
            float(r["distance_mae_all_pairs"]) >= args.min_mae or float(r["ca_rmsd_after_kabsch"]) >= args.min_rmsd,
            float(r["distance_mae_all_pairs"]),
            float(r["ca_rmsd_after_kabsch"]),
        ),
        reverse=True,
    )

    summary = args.out_dir / "summary.tsv"
    fields = [
        "rank", "name", "pdb_id", "length_eval", "distance_mae_all_pairs",
        "ca_rmsd_after_kabsch", "distance_corr_all_pairs", "mean_plddt",
        "evidence", "title",
    ]
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(out_rows, 1):
            writer.writerow({**{k: row.get(k, "") for k in fields}, "rank": rank})

    selected = [
        row for row in out_rows
        if float(row["distance_mae_all_pairs"]) >= args.min_mae or float(row["ca_rmsd_after_kabsch"]) >= args.min_rmsd
    ][: args.select_n]
    if len(selected) < args.select_n:
        selected = out_rows[: args.select_n]
    for row in selected:
        row["class"] = "hard_synthetic_miss"
        row["evidence"] = f"{row.get('evidence', '')};prescreen_mae={float(row['distance_mae_all_pairs']):.2f};prescreen_rmsd={float(row['ca_rmsd_after_kabsch']):.2f};prescreen_plddt={float(row['mean_plddt']):.3f}"
    write_manifest(args.selected_manifest, selected)
    print(f"wrote {summary}")
    print(f"wrote {args.selected_manifest}")


if __name__ == "__main__":
    main()
