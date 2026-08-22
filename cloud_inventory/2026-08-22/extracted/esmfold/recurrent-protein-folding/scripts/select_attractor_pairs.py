#!/usr/bin/env python3
"""Select protein pairs and orphans for the attractor-basin steering experiment.

Pairs: proteins A, B where TM(pred_A, pred_B) > TM_PAIR_THRESH
       AND sequence identity < SEQ_ID_MAX (structurally similar, sequence-diverse).

Orphans: proteins with max TM to any other protein < TM_ORPHAN_MAX
         (no structural neighbour → isolated attractor basin).

Outputs a JSON: {pairs: [[nameA, nameB, tm], ...], orphans: [name, ...]}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "runs/general_convergence_ood500_native_cw30_stream"

TM_PAIR_THRESH  = 0.65   # minimum inter-protein TM to count as "similar"
TM_ORPHAN_MAX   = 0.50   # maximum inter-protein TM to count as "orphan"
SEQ_ID_MAX      = 0.40   # reject pairs that are too sequence-similar
MIN_L, MAX_L    = 60, 180  # length range for manageable compute


# ── geometry helpers ──────────────────────────────────────────────────────────
def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(0); tc = target - target.mean(0)
    h = pc.T @ tc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1., 1., det]) @ u.T).T + target.mean(0)


def tm_score(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 10: return 0.0
    a, b = a[:n], b[:n]
    aligned = kabsch_align(a, b)
    d = np.sqrt(((aligned - b) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def seq_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0: return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir",      type=Path, default=OOD_DIR)
    ap.add_argument("--out",          type=Path, default=OOD_DIR / "attractor_pairs.json")
    ap.add_argument("--n-pairs",      type=int,  default=10)
    ap.add_argument("--n-orphans",    type=int,  default=10)
    ap.add_argument("--tm-pair",      type=float, default=TM_PAIR_THRESH)
    ap.add_argument("--tm-orphan",    type=float, default=TM_ORPHAN_MAX)
    ap.add_argument("--seq-id-max",   type=float, default=SEQ_ID_MAX)
    ap.add_argument("--min-len",      type=int,  default=MIN_L)
    ap.add_argument("--max-len",      type=int,  default=MAX_L)
    args = ap.parse_args()

    # ── collect candidates ────────────────────────────────────────────────────
    proteins = {}
    for d in sorted(args.ood_dir.glob("*_analysis")):
        name = d.name.removesuffix("_analysis")
        pred_p = d / "esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_p   = d / "run_artifacts/gt_ca_distances.npz"
        src_p  = d / "source.json"
        if not (pred_p.exists() and gt_p.exists() and src_p.exists()):
            continue
        pred_ca = np.load(pred_p)["pred_ca"].astype(float)
        gt_data = np.load(gt_p)
        gt_ca   = gt_data["ca_coords"].astype(float)
        seq     = str(gt_data.get("sequence", b""))
        if not seq:
            src = json.loads(src_p.read_text())
            seq = src.get("sequence", "")
        L = len(pred_ca)
        if not (args.min_len <= L <= args.max_len):
            continue
        proteins[name] = {"pred_ca": pred_ca, "gt_ca": gt_ca, "seq": seq, "L": L}

    names = sorted(proteins)
    N = len(names)
    print(f"Candidates in [{args.min_len}, {args.max_len}]: {N}")

    # ── pairwise TM on predicted structures ───────────────────────────────────
    print("Computing pairwise TM-scores (predicted structures)...")
    tm_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            t = tm_score(proteins[names[i]]["pred_ca"], proteins[names[j]]["pred_ca"])
            tm_mat[i, j] = tm_mat[j, i] = t
        if i % 20 == 0:
            print(f"  {i}/{N}")

    # ── select pairs ──────────────────────────────────────────────────────────
    pairs = []
    used  = set()
    # Sort by TM descending to pick best matches first
    candidates = [(tm_mat[i, j], i, j)
                  for i in range(N) for j in range(i + 1, N)
                  if tm_mat[i, j] >= args.tm_pair]
    candidates.sort(reverse=True)
    for tm_val, i, j in candidates:
        if len(pairs) >= args.n_pairs:
            break
        ni, nj = names[i], names[j]
        if ni in used or nj in used:
            continue
        si = seq_identity(proteins[ni]["seq"], proteins[nj]["seq"])
        if si > args.seq_id_max:
            continue
        pairs.append({"name_a": ni, "name_b": nj,
                      "tm_pred": round(float(tm_val), 4),
                      "seq_id":  round(float(si), 3),
                      "L_a": proteins[ni]["L"], "L_b": proteins[nj]["L"]})
        used.update([ni, nj])
        print(f"  pair: {ni} ↔ {nj}  TM={tm_val:.3f}  seqid={si:.3f}")

    # ── select orphans ────────────────────────────────────────────────────────
    orphans = []
    max_tms = {names[i]: float(np.max(np.concatenate([tm_mat[i, :i], tm_mat[i, i+1:]])))
               for i in range(N)}
    orphan_cands = sorted(
        [(v, k) for k, v in max_tms.items() if k not in used and v <= args.tm_orphan],
        key=lambda x: x[0]
    )
    for max_tm, name in orphan_cands[:args.n_orphans]:
        orphans.append({"name": name, "max_tm_to_others": round(float(max_tm), 4),
                        "L": proteins[name]["L"]})
        print(f"  orphan: {name}  max_TM={max_tm:.3f}")

    result = {"pairs": pairs, "orphans": orphans,
              "n_candidates": N, "tm_pair_thresh": args.tm_pair,
              "tm_orphan_max": args.tm_orphan}
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    print(f"pairs={len(pairs)}  orphans={len(orphans)}")


if __name__ == "__main__":
    main()
