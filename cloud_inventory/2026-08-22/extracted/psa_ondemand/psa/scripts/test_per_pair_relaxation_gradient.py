"""Test that per-pair editing at block b+1 is gradient-following on the per-pair
non-Euclidean violation at block b.

For each block b, each val protein:
  1. D_b   = predicted Cα distance map at block b (via cached distogram-head output)
  2. Compute best rank-3 Euclidean embedding via classical MDS:
       M = -1/2 J D_b² J,  positive eigenvalues + top-3 eigenvectors → coords X (L, 3)
       D_b^(3) = pairwise distances of X
  3. Per-pair non-Euclidean violation:  V_b[i, j] = |D_b[i, j] − D_b^(3)[i, j]|
     (how far this pair's distance is from being consistent with a rank-3 embedding)
  4. Per-pair next-step edit:  edit_b[i, j] = |D_{b+1}[i, j] − D_b[i, j]|

Prediction (SDP-solver-like behavior):
  Per-pair |V_b| should POSITIVELY correlate with |edit_b| — pairs that violate
  rank-3 embedding most get the next step's editing pressure. This should be
  strongest during the non-metric phase.

We also residualize by distance (gt_dist) to make sure the correlation isn't
just a distance proxy.

Outputs:
  runs/relaxation_gradient_test.json   per-block correlations
  runs/relaxation_gradient.png          per-block correlation curves
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
log = logging.getLogger("relax_gradient")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
DISTO_DIR = ROOT / "data" / "distograms"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

N_BLOCKS = 48
SEED = 0


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def mds_rank_k_reconstruction(D: np.ndarray, k: int = 3) -> np.ndarray:
    """Classical MDS: best rank-k Euclidean reconstruction of distance matrix D.
    Returns reconstructed distance matrix D_hat of same shape."""
    L = D.shape[0]
    J = np.eye(L) - np.ones((L, L)) / L
    M = -0.5 * J @ (D ** 2) @ J
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)        # ascending order
    # Pick top-k POSITIVE eigenvalues
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    pos = eigvals > 0
    keep = np.where(pos)[0][:k]
    if len(keep) == 0:
        return np.zeros_like(D)
    lam = eigvals[keep]
    V = eigvecs[:, keep]
    X = V * np.sqrt(lam)[None, :]                # (L, k)
    # Reconstructed distances
    diff = X[:, None, :] - X[None, :, :]
    D_hat = np.sqrt((diff ** 2).sum(-1))
    return D_hat


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def residualize(y: np.ndarray, x: np.ndarray, deg: int = 3) -> np.ndarray:
    """Return y − polyfit(x, y) — residual after regressing on x."""
    coef = np.polyfit(x, y, deg=deg)
    return y - np.polyval(coef, x)


def main() -> None:
    np.random.seed(SEED); random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows
                if (DISTO_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    # Load proteins
    proteins = []
    for pid in val_ids:
        d = torch.load(DISTO_DIR / f"{pid.lower()}.pt",
                        map_location="cpu", weights_only=False)
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        L = d["L"]
        if phys["distances"].shape[0] != L:
            continue
        proteins.append({"pdb_id": pid, "L": L,
                          "pred": d["pred_distance"].float().numpy(),
                          "gt": phys["distances"].astype(np.float32)})
    log.info("Loaded %d proteins", len(proteins))

    # Per-block correlations (pooled across val proteins)
    per_block = {
        "pearson_violation_vs_next_edit": [None] * (N_BLOCKS - 1),
        "pearson_violation_vs_next_edit_resid_d": [None] * (N_BLOCKS - 1),
        "pearson_violation_vs_5step_future_edit": [None] * (N_BLOCKS - 1),
        "mean_violation": [None] * N_BLOCKS,
    }

    # Pre-compute |violation| per block per protein
    log.info("Computing rank-3 MDS reconstructions per block per protein…")
    violations = {}    # {pid: (n_blocks, L, L) absolute violation matrix}
    for p in proteins:
        L = p["L"]
        V = np.zeros((N_BLOCKS, L, L), dtype=np.float32)
        for b in range(N_BLOCKS):
            D = p["pred"][b].astype(np.float64)
            D_hat = mds_rank_k_reconstruction(D, k=3)
            V[b] = np.abs(D - D_hat).astype(np.float32)
        violations[p["pdb_id"]] = V

    # Per-block aggregate
    for b in range(N_BLOCKS):
        per_block["mean_violation"][b] = float(
            np.mean([violations[p["pdb_id"]][b][~np.eye(p["L"], dtype=bool)].mean()
                     for p in proteins])
        )

    log.info("Computing per-block correlations…")
    for b in range(N_BLOCKS - 1):
        v_pool, e_pool, e5_pool, d_pool = [], [], [], []
        for p in proteins:
            L = p["L"]
            mask = ~np.eye(L, dtype=bool)
            V_b = violations[p["pdb_id"]][b][mask]
            edit_next = np.abs(p["pred"][b + 1] - p["pred"][b])[mask]
            # Sum of edits over next 5 blocks (if available)
            b_end = min(b + 6, N_BLOCKS)
            edit_5 = np.zeros_like(V_b)
            for b2 in range(b, b_end - 1):
                edit_5 = edit_5 + np.abs(p["pred"][b2 + 1] - p["pred"][b2])[mask]
            v_pool.append(V_b)
            e_pool.append(edit_next)
            e5_pool.append(edit_5)
            d_pool.append(p["gt"][mask])
        V_b = np.concatenate(v_pool)
        e = np.concatenate(e_pool)
        e5 = np.concatenate(e5_pool)
        d = np.concatenate(d_pool)
        per_block["pearson_violation_vs_next_edit"][b] = pearson(V_b, e)
        # Residualize both by distance, then correlate
        Vr = residualize(V_b, d, deg=3)
        er = residualize(e, d, deg=3)
        per_block["pearson_violation_vs_next_edit_resid_d"][b] = pearson(Vr, er)
        per_block["pearson_violation_vs_5step_future_edit"][b] = pearson(V_b, e5)

    # Save
    (RUNS_DIR / "relaxation_gradient_test.json").write_text(json.dumps(per_block, indent=2))
    log.info("Wrote runs/relaxation_gradient_test.json")

    # Print summary
    log.info("\nPer-block correlations (selected blocks):")
    log.info("%6s %15s %20s %20s %15s",
              "block", "mean_|V|", "r(V→next_edit)", "r(V→next_edit|d)", "r(V→5step_edit)")
    for b in [0, 5, 10, 15, 20, 22, 25, 28, 32, 36, 40, 42, 43, 44, 45, 46]:
        log.info("%6d %15.3f %20.4f %20.4f %15.4f", b,
                  per_block["mean_violation"][b],
                  per_block["pearson_violation_vs_next_edit"][b] or 0,
                  per_block["pearson_violation_vs_next_edit_resid_d"][b] or 0,
                  per_block["pearson_violation_vs_5step_future_edit"][b] or 0)

    # Plot
    t = np.arange(N_BLOCKS - 1) + 0.5
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    ax.plot(np.arange(N_BLOCKS), per_block["mean_violation"], "-o",
             color="C5", markersize=3, label="mean |V| (rank-3 MDS error)")
    ax.set_ylabel("mean |D − D_hat_rank3| (Å)")
    ax.set_title("Per-block non-Euclidean violation magnitude")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    r_raw = np.array(per_block["pearson_violation_vs_next_edit"])
    r_resid = np.array(per_block["pearson_violation_vs_next_edit_resid_d"])
    r_5step = np.array(per_block["pearson_violation_vs_5step_future_edit"])
    ax.plot(t, r_raw, "-o", color="C0", markersize=3, label="r(|V_b|, next |edit|)")
    ax.plot(t, r_resid, "-s", color="C2", markersize=3,
             label="r(|V_b|, next |edit|) — residualized for distance")
    ax.plot(t, r_5step, "-^", color="C3", markersize=3, alpha=0.6,
             label="r(|V_b|, 5-step future |edit|)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("block (transition b → b+1 plotted at b+0.5)")
    ax.set_ylabel("Pearson r")
    ax.set_title("SDP-gradient prediction: do high-violation pairs get edited next?")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("runs/relaxation_gradient.png", dpi=130, bbox_inches="tight")
    log.info("saved runs/relaxation_gradient.png")


if __name__ == "__main__":
    main()
