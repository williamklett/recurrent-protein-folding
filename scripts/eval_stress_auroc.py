#!/usr/bin/env python3
"""Compute mean MDS stress per protein across all layers, compare AUROC vs pLDDT.

Uses cached esmfold_activations.pt (no ESMFold re-run needed).
Probes from general_belief_replay_benchmark20_cw30.

Output: stress_auroc_results.csv  +  stress_auroc_comparison.png
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"
OUT_DIR    = ROOT / "runs"

FIRST_LAYER = 1
LAST_LAYER  = 46


# ── geometry ──────────────────────────────────────────────────────────────────
def kabsch_tm(pred: np.ndarray, gt: np.ndarray) -> float:
    n = min(len(pred), len(gt))
    p, g = pred[:n].copy(), gt[:n].copy()
    p -= p.mean(0); gc = g - g.mean(0)
    H = p.T @ gc; U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    pa = p @ R.T + g.mean(0)
    d  = np.sqrt(((pa - g) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1 / (1 + (d / d0) ** 2)))


def classical_mds(D: np.ndarray, k: int = 3) -> np.ndarray:
    n = D.shape[0]
    D2 = D ** 2
    J  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx  = np.argsort(-vals)
    vals = np.maximum(vals[idx[:k]], 0)
    vecs = vecs[:, idx[:k]]
    return vecs * np.sqrt(vals)


def mds_stress(D: np.ndarray) -> float:
    coords = classical_mds(D)
    D_emb  = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    mask   = np.triu(np.ones(D.shape, dtype=bool), 1)
    return float(np.sqrt(np.mean((D_emb[mask] - D[mask]) ** 2)))


# ── probe ─────────────────────────────────────────────────────────────────────
def load_probe(layer: int, device: torch.device) -> PairDistogramBottleneck:
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(
        pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256
    ).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def expected_dist(z_layer: torch.Tensor, probe, centers: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    pairs = z_layer[ii_f, jj_f].to(device)
    ii_f, jj_f = ii_f.to(device), jj_f.to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_f, jj_f, L)
        probs  = F.softmax(logits, dim=-1)
        ed     = (probs * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy()


# ── data ──────────────────────────────────────────────────────────────────────
def find_proteins() -> list[dict]:
    results = []
    for act_p in sorted(RUNS_DIR.glob("*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p    = run_dir / "gt_ca_distances.npz"
        pred_p  = next(run_dir.glob("**/esmfold_prediction_ca.npz"), None)
        if not gt_p.exists() or pred_p is None:
            continue
        results.append({"name": run_dir.name, "act": act_p,
                         "gt": gt_p, "pred": pred_p})
    return results


# ── AUROC ─────────────────────────────────────────────────────────────────────
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    auc = tpr = fpr = prev_fpr = 0.0
    for i in order:
        if labels[i]: tpr += 1 / n_pos
        else:
            fpr += 1 / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr
    return float(auc)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-csv", type=Path, default=OUT_DIR / "stress_auroc_results.csv")
    ap.add_argument("--out-png", type=Path, default=OUT_DIR / "stress_auroc_comparison.png")
    args = ap.parse_args()

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32)

    proteins = find_proteins()
    print(f"Found {len(proteins)} proteins with cached activations")

    rows = []
    for pi, prot in enumerate(proteins):
        name = prot["name"]
        print(f"[{pi+1}/{len(proteins)}] {name}", flush=True)

        # load cached z
        act   = torch.load(prot["act"], map_location="cpu", weights_only=False)
        z_all = act["z"].squeeze(1).float()   # (48, L, L, 128)

        # TM score
        gt_data = np.load(prot["gt"])
        gt_ca   = gt_data["ca_coords"].astype(float)
        pred_ca = np.load(prot["pred"])["pred_ca"].astype(float)
        tm      = kabsch_tm(pred_ca, gt_ca)

        # mean MDS stress across layers
        stresses = []
        for t in range(FIRST_LAYER, LAST_LAYER + 1):
            probe = load_probe(t, device)
            D = expected_dist(z_all[t], probe, centers, device)
            stresses.append(mds_stress(D))
            del probe
        torch.cuda.empty_cache()

        mean_stress = float(np.mean(stresses))
        max_stress  = float(np.max(stresses))
        print(f"  TM={tm:.3f}  mean_stress={mean_stress:.2f}  max_stress={max_stress:.2f}")
        rows.append({"name": name, "tm": tm,
                     "mean_stress": mean_stress, "max_stress": max_stress})

    # save CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "tm", "mean_stress", "max_stress"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out_csv}")

    # AUROC
    tm_arr     = np.array([r["tm"] for r in rows])
    stress_arr = np.array([r["mean_stress"] for r in rows])
    fail       = tm_arr < 0.5
    auc_stress = auroc(stress_arr, fail)   # higher stress = failure
    print(f"\nN={len(rows)}  failures(TM<0.5)={fail.sum()}  successes={( ~fail).sum()}")
    print(f"AUROC  mean_stress={auc_stress:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.12)
    fig.suptitle("MDS stress vs TM-score",
                 x=0.11, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color="#172033")
    fig.text(0.11, 0.88,
             f"Mean stress across layers 1–46 from cw30 probes.  "
             f"AUROC={auc_stress:.3f}  N={len(rows)}  "
             f"(failures TM<0.5: {fail.sum()})",
             ha="left", fontsize=10, color="#475569")

    colors = ["#ef4444" if r["tm"] < 0.5 else "#16a34a" for r in rows]
    ax.scatter(stress_arr, tm_arr, c=colors, s=80,
               edgecolors="#172033", lw=0.6, zorder=3)
    ax.axhline(0.5, color="#94a3b8", lw=1.2, ls="--", alpha=0.6)
    for r, c in zip(rows, colors):
        ax.annotate(r["name"].replace("benchmark_", "").replace("hard_synthetic_miss_", "✗"),
                    (r["mean_stress"], r["tm"]),
                    fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Mean MDS stress across layers (Å)", fontsize=12)
    ax.set_ylabel("TM-score vs GT", fontsize=12)
    ax.set_title(f"Higher stress → worse fold   AUROC={auc_stress:.3f}", fontsize=12)
    ax.grid(True, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(args.out_png, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
