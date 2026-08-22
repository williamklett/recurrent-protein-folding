"""Direct test of the 'soft SDP relaxation' hypothesis:

For an embeddable (metric, R³-realizable) distance matrix D, the doubly-centered
squared-distance matrix
    M = -1/2 * J D² J,    where J = I − (1/L) 1 1ᵀ
is positive semi-definite (PSD) with rank ≤ 3. The spectrum of M is:

  - embeddable metric    →  top-3 large positive eigenvalues, rest ≈ 0, no negative
  - non-metric           →  some negative eigenvalues appear (M is no longer PSD)
  - SDP-relaxation-style →  multiple positive eigenvalues AND negative ones (rank
                            broken, PSD broken)

We apply ESMFold's own distogram head to every block's pair_rep, get predicted
distance maps per (block, protein), compute M's eigenvalue spectrum, and look at:

  * positive_energy(b)        sum of positive eigenvalues
  * top3_fraction(b)          top-3 fraction of positive energy
  * effective_rank(b)         number of eigenvalues with |λ| > threshold
  * negative_energy(b)        sum of |negative eigenvalues|
  * neg_to_pos_ratio(b)       negative_energy / positive_energy
  * worst_neg_eigval(b)       most negative eigenvalue (magnitude)

Prediction (lift-and-project):
  * Phase 1 (blocks 5–20):   no negatives, rank ≈ 3, top-3 dominates
  * Phase 2 (blocks 25–40):  systematic negatives, rank inflated, top-3 leaks
  * Phase 4 (blocks 43–47):  back to no negatives, rank ≈ 3

If we see negative eigenvalues that *grow and shrink with the M-shape of triangle
violations*, that's strong evidence for the SDP-relaxation specifically.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("sdp_test")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
DISTO_DIR = ROOT / "data" / "distograms"
RUNS_DIR = ROOT / "runs"

N_BLOCKS = 48
SEED = 0


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def centering(L: int) -> np.ndarray:
    return np.eye(L) - np.ones((L, L)) / L


def double_center_eigvals(D: np.ndarray) -> np.ndarray:
    """Return sorted (descending) eigenvalues of -1/2 J D² J."""
    L = D.shape[0]
    J = centering(L)
    D2 = D ** 2
    M = -0.5 * J @ D2 @ J
    M = 0.5 * (M + M.T)        # enforce symmetry, kill numerical asymmetry
    eigvals = np.linalg.eigvalsh(M)         # ascending order
    return eigvals[::-1].astype(np.float32)  # descending


def main() -> None:
    np.random.seed(SEED); random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows
                if (DISTO_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    proteins = []
    for pid in val_ids:
        d = torch.load(DISTO_DIR / f"{pid.lower()}.pt",
                        map_location="cpu", weights_only=False)
        proteins.append({"pdb_id": pid, "L": d["L"],
                          "pred": d["pred_distance"].float().numpy()})
    log.info("Loaded %d proteins", len(proteins))

    # Per (block, protein) → eigenvalue stats
    # Storage: per-block arrays aggregated across proteins
    per_block_stats = {
        "positive_energy": [], "top3_fraction": [], "effective_rank": [],
        "negative_energy": [], "neg_to_pos_ratio": [],
        "worst_neg_eigval": [], "neg_count_above_thresh": [],
        "top5_per_block": [],   # full top-5 eigenvalue trajectory per protein
        "neg5_per_block": [],   # full top-5 magnitude negative
    }

    # For ground truth check: compute on first val protein's GT distance
    log.info("\nSanity-check eigenvalue spectrum of GROUND-TRUTH distance map:")
    phys = np.load(ROOT / "data" / "physics" / f"{val_ids[0].lower()}.npz")
    gt = phys["distances"].astype(np.float64)
    eig_gt = double_center_eigvals(gt)
    top5_gt = eig_gt[:5]
    neg_gt = eig_gt[eig_gt < 0]
    log.info("  %s ground truth:  top-5 = %s", val_ids[0],
              ["%.1f" % v for v in top5_gt])
    log.info("                    n_neg = %d, sum |neg| = %.2f, frac_neg/pos = %.3f",
              len(neg_gt),
              np.abs(neg_gt).sum() if len(neg_gt) else 0.0,
              np.abs(neg_gt).sum() / eig_gt[eig_gt > 0].sum())

    # Per-block stats
    for b in range(N_BLOCKS):
        pos_e, neg_e, top3f, eff_rank = [], [], [], []
        neg_to_pos, worst_neg, n_neg_above = [], [], []
        top5_all, neg5_all = [], []
        for p in proteins:
            D = p["pred"][b].astype(np.float64)
            eig = double_center_eigvals(D)
            pos = eig[eig > 0]
            neg = eig[eig < 0]
            pe = float(pos.sum())
            ne = float(np.abs(neg).sum()) if len(neg) else 0.0
            pos_e.append(pe); neg_e.append(ne)
            top3f.append(float(pos[:3].sum() / max(pe, 1e-9)))
            # Effective rank: count eigenvalues whose |val| > 1% of max
            thresh = 0.01 * float(np.abs(eig).max())
            eff_rank.append(int((np.abs(eig) > thresh).sum()))
            neg_to_pos.append(ne / max(pe, 1e-9))
            worst_neg.append(float(eig.min()) if len(neg) else 0.0)
            n_neg_above.append(int((np.abs(eig) > thresh).sum() & (eig < 0).sum()))
            # Top-5 positive and top-5 magnitude negative
            top5 = list(eig[:5])
            while len(top5) < 5:
                top5.append(0.0)
            top5_all.append(top5)
            neg_sorted = np.sort(eig)
            neg5 = list(neg_sorted[:5])     # most negative first
            while len(neg5) < 5:
                neg5.append(0.0)
            neg5_all.append(neg5)
        per_block_stats["positive_energy"].append(float(np.mean(pos_e)))
        per_block_stats["top3_fraction"].append(float(np.mean(top3f)))
        per_block_stats["effective_rank"].append(float(np.mean(eff_rank)))
        per_block_stats["negative_energy"].append(float(np.mean(neg_e)))
        per_block_stats["neg_to_pos_ratio"].append(float(np.mean(neg_to_pos)))
        per_block_stats["worst_neg_eigval"].append(float(np.mean(worst_neg)))
        per_block_stats["top5_per_block"].append(np.array(top5_all).mean(0).tolist())
        per_block_stats["neg5_per_block"].append(np.array(neg5_all).mean(0).tolist())

    # Save
    (RUNS_DIR / "sdp_relaxation_test.json").write_text(json.dumps(per_block_stats, indent=2))
    log.info("Wrote runs/sdp_relaxation_test.json")

    # Print key trajectory
    log.info("\n=== Per-block summary (mean over %d val proteins) ===", len(proteins))
    log.info("%6s %10s %10s %10s %10s %12s %10s",
              "block", "top3_frac", "pos_E", "neg_E", "neg/pos", "eff_rank", "worst_neg")
    for b in [0, 5, 10, 15, 20, 22, 25, 28, 32, 36, 40, 42, 43, 44, 45, 46, 47]:
        log.info("%6d %10.3f %10.1f %10.1f %10.3f %12.1f %10.1f", b,
                  per_block_stats["top3_fraction"][b],
                  per_block_stats["positive_energy"][b],
                  per_block_stats["negative_energy"][b],
                  per_block_stats["neg_to_pos_ratio"][b],
                  per_block_stats["effective_rank"][b],
                  per_block_stats["worst_neg_eigval"][b])

    # Plots
    b_arr = np.arange(N_BLOCKS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(b_arr, per_block_stats["top3_fraction"], "-o", color="C0", markersize=3)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="top-3 = 100% (true 3D)")
    ax.set_ylabel("top-3 fraction of positive energy"); ax.set_title("1. Rank-3 dominance — does top-3 carry all the metric info?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(b_arr, per_block_stats["negative_energy"], "-o", color="C3", markersize=3, label="Σ|neg eigvals|")
    ax.plot(b_arr, per_block_stats["positive_energy"], "-o", color="C2", markersize=3, alpha=0.5, label="Σ pos eigvals")
    ax.set_ylabel("eigenvalue energy"); ax.set_title("2. Positive vs negative energy in M = −½ J D² J")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(b_arr, per_block_stats["neg_to_pos_ratio"], "-o", color="C4", markersize=3)
    ax.set_xlabel("block"); ax.set_ylabel("Σ|neg| / Σpos")
    ax.set_title("3. Non-embeddability index — ratio of negative to positive energy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(b_arr, per_block_stats["effective_rank"], "-o", color="C5", markersize=3, label="effective rank")
    ax.axhline(3, color="gray", ls="--", alpha=0.5, label="rank-3 (true 3D embedding)")
    ax.set_xlabel("block"); ax.set_ylabel("effective rank (|λ| > 1% of max)")
    ax.set_title("4. Effective rank — does it inflate above 3 during the relaxed phase?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Lift-and-project test: spectrum of the doubly-centered squared-distance matrix per block",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig("runs/sdp_relaxation.png", dpi=130, bbox_inches="tight")
    log.info("saved runs/sdp_relaxation.png")

    # Detail: top-5 positive + top-5 negative eigenvalues trajectory
    top5 = np.array(per_block_stats["top5_per_block"])
    neg5 = np.array(per_block_stats["neg5_per_block"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    for i in range(5):
        ax.plot(b_arr, top5[:, i], "-o", markersize=3, label=f"λ_{i+1} (pos)")
    ax.set_ylabel("eigenvalue (positive side, top-5)")
    ax.set_title("Top-5 positive eigenvalues of M per block")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(5):
        ax.plot(b_arr, neg5[:, i], "-o", markersize=3, label=f"λ_{-i-1} (neg)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("block"); ax.set_ylabel("eigenvalue (most negative, top-5)")
    ax.set_title("Most-negative-5 eigenvalues per block (zero ⇒ PSD ⇒ embeddable)")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/sdp_relaxation_eigvals.png", dpi=130, bbox_inches="tight")
    log.info("saved runs/sdp_relaxation_eigvals.png")


if __name__ == "__main__":
    main()
