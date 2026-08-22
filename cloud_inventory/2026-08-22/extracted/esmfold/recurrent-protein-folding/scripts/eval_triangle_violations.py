#!/usr/bin/env python3
"""Triangle inequality violation rate per protein, averaged across layers.

For each layer, get E[d] from probe, sample N_TRIPLES random triples,
check fraction violating d(i,k) > d(i,j) + d(j,k).
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"
N_TRIPLES  = 3000
FIRST, LAST = 1, 46


def load_probe(layer, device):
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters(): p.requires_grad_(False)
    return pr


def get_ed(z_layer, probe, centers, device):
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    pairs = z_layer[ii_f, jj_f].to(device)
    ii_d, jj_d = ii_f.to(device), jj_f.to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_d, jj_d, L)
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy()


def triangle_violation_rate(D, n, rng):
    L = D.shape[0]
    idx = rng.integers(0, L, size=(n * 2, 3))
    mask = (idx[:,0] != idx[:,1]) & (idx[:,1] != idx[:,2]) & (idx[:,0] != idx[:,2])
    idx = idx[mask][:n]
    if len(idx) == 0: return float("nan")
    i, j, k = idx[:,0], idx[:,1], idx[:,2]
    v = (D[i,k] > D[i,j] + D[j,k]) | \
        (D[i,j] > D[i,k] + D[k,j]) | \
        (D[j,k] > D[j,i] + D[i,k])
    return float(v.mean())


def kabsch_tm(pred, gt):
    n = min(len(pred), len(gt))
    p, g = pred[:n].copy(), gt[:n].copy()
    p -= p.mean(0); gc = g - g.mean(0)
    H = p.T @ gc; U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1,1,np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    pa = p @ R.T + g.mean(0)
    d = np.sqrt(((pa - g)**2).sum(-1))
    d0 = max(0.5, 1.24*np.cbrt(max(n-15,1)) - 1.8)
    return float(np.mean(1/(1+(d/d0)**2)))


def auroc(scores, labels):
    n_pos, n_neg = labels.sum(), len(labels)-labels.sum()
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(-scores)
    auc = tpr = fpr = prev = 0.0
    for i in order:
        if labels[i]: tpr += 1/n_pos
        else: fpr += 1/n_neg; auc += tpr*(fpr-prev); prev=fpr
    return float(auc)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32)
    rng = np.random.default_rng(42)

    proteins = []
    for act_p in sorted(RUNS_DIR.glob("*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p  = run_dir / "gt_ca_distances.npz"
        pred_p = next(run_dir.glob("**/esmfold_prediction_ca.npz"), None)
        if gt_p.exists() and pred_p:
            proteins.append((run_dir.name, act_p, gt_p, pred_p))

    print(f"Found {len(proteins)} proteins")
    rows = []

    for pi, (name, act_p, gt_p, pred_p) in enumerate(proteins):
        print(f"[{pi+1}/{len(proteins)}] {name}", flush=True)
        act   = torch.load(act_p, map_location="cpu", weights_only=False)
        z_all = act["z"].squeeze(1).float()
        gt_ca   = np.load(gt_p)["ca_coords"].astype(float)
        pred_ca = np.load(pred_p)["pred_ca"].astype(float)
        tm = kabsch_tm(pred_ca, gt_ca)

        viols = []
        for t in range(FIRST, LAST + 1):
            probe = load_probe(t, device)
            D = get_ed(z_all[t], probe, centers, device)
            viols.append(triangle_violation_rate(D, N_TRIPLES, rng))
            del probe
        torch.cuda.empty_cache()

        mean_viol = float(np.mean(viols))
        max_viol  = float(np.max(viols))
        print(f"  TM={tm:.3f}  mean_viol={mean_viol:.4f}  max_viol={max_viol:.4f}")
        rows.append({"name": name, "tm": tm,
                     "mean_viol": mean_viol, "max_viol": max_viol})

    tm_arr   = np.array([r["tm"]        for r in rows])
    mv_arr   = np.array([r["mean_viol"] for r in rows])
    xv_arr   = np.array([r["max_viol"]  for r in rows])
    fail     = tm_arr < 0.5

    print(f"\nN={len(rows)}  failures={fail.sum()}")
    print(f"AUROC mean_viol={auroc(mv_arr, fail):.4f}")
    print(f"AUROC max_viol ={auroc(xv_arr, fail):.4f}")

    out = ROOT / "runs/triangle_violation_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","tm","mean_viol","max_viol"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
