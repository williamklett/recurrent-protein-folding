"""Direct (p, q) signature analysis of the trunk's decoded distance maps.

For each block's predicted distance map D, compute the doubly-centered matrix
M = -½ J D² J and its full eigenvalue spectrum. Define:

  p_eff(b, threshold) = # eigenvalues > +threshold × max|λ|
  q_eff(b, threshold) = # eigenvalues < −threshold × max|λ|

Then the pseudo-Euclidean signature realizing the decoded distances is (p_eff,
q_eff) with embedding dimension p_eff + q_eff.

A strict-Euclidean rank-3 distance map has p_eff = 3, q_eff = 0.
SDP-relaxation (PSD with high rank) would have p_eff ≫ 3, q_eff = 0.
Hyperbolic / Minkowski-(p, 1) would have p_eff ≫ 0, q_eff = 1.
Generic indefinite (our claim) has p_eff ≫ 0, q_eff ≫ 0.

We also do a *direct* pseudo-Euclidean fit: for each q ∈ {0, 1, 2, 4, 8, 16, 32},
fit per-residue (x⁺ ∈ ℝᵖ, x⁻ ∈ ℝᑫ) coordinates via classical pseudo-Euclidean
MDS (which is essentially the same as eigendecomposition) and measure
reconstruction stress. Find the q where adding more negative dimensions stops
improving fit — that's the *required* q for embedding the distance map.

Output:
  runs/pq_signature_summary.json         per-block p_eff/q_eff + per-protein values
  runs/pq_signature_headline.png         p_eff(block), q_eff(block), q_eff/p_eff ratio
  runs/pq_fit_quality.png                reconstruction stress vs q at selected blocks
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
log = logging.getLogger("pq")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
DISTO_DIR = ROOT / "data" / "distograms"
RUNS_DIR = ROOT / "runs"

N_BLOCKS = 48
SEED = 0
TAU = 0.01   # threshold: keep eigenvalues with |λ| > τ × max|λ|

# For direct (p, q) fit
Q_VALUES = [0, 1, 2, 4, 8, 16, 32, 64]
P_FIXED = 16    # fix p, scan q


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def pq_decomposition(D: np.ndarray):
    """Return (eigvals sorted descending, eigvecs).  M = U Λ Uᵀ."""
    L = D.shape[0]
    J = np.eye(L) - np.ones((L, L)) / L
    M = -0.5 * J @ (D ** 2) @ J
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)   # ascending
    return eigvals[::-1], eigvecs[:, ::-1]


def p_q_effective(eigvals: np.ndarray, tau: float = TAU) -> tuple[int, int]:
    max_abs = float(np.abs(eigvals).max()) if len(eigvals) else 1.0
    thr = tau * max_abs
    p_eff = int((eigvals > thr).sum())
    q_eff = int((eigvals < -thr).sum())
    return p_eff, q_eff


def pseudo_euclidean_recon_error(eigvals: np.ndarray, eigvecs: np.ndarray,
                                   p: int, q: int) -> float:
    """Reconstruct D using top-p positive eigvals + top-q magnitude negative.
    Return relative reconstruction error of D² (Frobenius).

    With pseudo-Euclidean coords x⁺ ∈ ℝᵖ, x⁻ ∈ ℝᑫ:
      d²_ij = ||x⁺ᵢ − x⁺ⱼ||² − ||x⁻ᵢ − x⁻ⱼ||² + offset_correction
    Equivalent to keeping top-p positive and top-q negative eigvalues of M.
    """
    L = len(eigvals)
    pos_idx = np.where(eigvals > 0)[0][:p] if p > 0 else np.array([], dtype=int)
    neg_idx = np.where(eigvals < 0)[0]
    # Sort negatives by magnitude (already sorted by value descending so most-negative is last)
    if q > 0 and len(neg_idx) > 0:
        # eigvals are descending; negatives are at the END; most-negative is at end.
        # Take the q most-negative.
        neg_idx = neg_idx[-q:]
    else:
        neg_idx = np.array([], dtype=int)
    keep_idx = np.concatenate([pos_idx, neg_idx])
    if len(keep_idx) == 0:
        return 1.0
    Λ_keep = np.diag(eigvals[keep_idx])
    U_keep = eigvecs[:, keep_idx]
    M_recon = U_keep @ Λ_keep @ U_keep.T
    # Get back D²: D²_ij = M_ii + M_jj - 2 M_ij (under double-centering convention)
    diag = np.diag(M_recon)
    D2_recon = diag[:, None] + diag[None, :] - 2 * M_recon
    return D2_recon


def relative_fro_error(target: np.ndarray, recon: np.ndarray) -> float:
    diff = target - recon
    return float(np.linalg.norm(diff) / (np.linalg.norm(target) + 1e-9))


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

    # Per-block p_eff, q_eff aggregated across val proteins
    per_block = {"p_eff_mean": [], "q_eff_mean": [],
                  "p_eff_std": [], "q_eff_std": [],
                  "p_eff_per_protein": [], "q_eff_per_protein": []}

    log.info("Computing (p_eff, q_eff) per block per protein…")
    for b in range(N_BLOCKS):
        p_list, q_list = [], []
        for p in proteins:
            D = p["pred"][b].astype(np.float64)
            eigvals, _ = pq_decomposition(D)
            p_eff, q_eff = p_q_effective(eigvals, tau=TAU)
            p_list.append(p_eff)
            q_list.append(q_eff)
        per_block["p_eff_mean"].append(float(np.mean(p_list)))
        per_block["q_eff_mean"].append(float(np.mean(q_list)))
        per_block["p_eff_std"].append(float(np.std(p_list)))
        per_block["q_eff_std"].append(float(np.std(q_list)))
        per_block["p_eff_per_protein"].append(p_list)
        per_block["q_eff_per_protein"].append(q_list)

    log.info("\n=== Per-block (p_eff, q_eff) at selected blocks (mean ± std over val proteins) ===")
    log.info("%6s %12s %12s %14s", "block", "p_eff", "q_eff", "q_eff/p_eff")
    for b in [0, 5, 10, 15, 20, 22, 25, 28, 30, 32, 35, 40, 43, 44, 45, 46, 47]:
        p_m, p_s = per_block["p_eff_mean"][b], per_block["p_eff_std"][b]
        q_m, q_s = per_block["q_eff_mean"][b], per_block["q_eff_std"][b]
        ratio = q_m / max(p_m, 1)
        log.info("%6d  %6.1f±%-4.1f  %6.1f±%-4.1f  %12.3f",
                  b, p_m, p_s, q_m, q_s, ratio)

    # ===== Direct (p, q) fit at selected blocks =====
    log.info("\nDirect pseudo-Euclidean fit at selected blocks…")
    direct_fit = {}   # {block: {q: mean recon error across val proteins}}
    fit_blocks = [0, 15, 25, 35, 43, 47]
    for b in fit_blocks:
        errors_by_q = {q: [] for q in Q_VALUES}
        for p in proteins:
            D = p["pred"][b].astype(np.float64)
            eigvals, eigvecs = pq_decomposition(D)
            D2 = D ** 2
            for q in Q_VALUES:
                D2_recon = pseudo_euclidean_recon_error(eigvals, eigvecs, P_FIXED, q)
                err = relative_fro_error(D2, D2_recon)
                errors_by_q[q].append(err)
        direct_fit[b] = {q: float(np.mean(v)) for q, v in errors_by_q.items()}
        # Print
        msg = f"  block {b:3d}: "
        msg += "  ".join(f"q={q}:{direct_fit[b][q]:.3f}" for q in Q_VALUES)
        log.info(msg)

    # ===== L correlation with q_eff WITHIN each block =====
    log.info("\nq_eff vs L correlation per block (sharper version of §5.4):")
    L_vals = np.array([p["L"] for p in proteins])
    from scipy.stats import pearsonr
    q_l_corr = []
    for b in range(N_BLOCKS):
        q_vals = np.array(per_block["q_eff_per_protein"][b])
        if q_vals.std() < 0.5:
            q_l_corr.append(float("nan"))
            continue
        r, p_val = pearsonr(L_vals, q_vals)
        q_l_corr.append(r)
    for b in [5, 15, 22, 25, 30, 35, 43, 47]:
        log.info("  block %2d  r(L, q_eff) = %+.3f", b, q_l_corr[b])

    out = {
        "tau": TAU,
        "blocks": list(range(N_BLOCKS)),
        "per_block": per_block,
        "direct_fit_p_fixed": P_FIXED,
        "direct_fit_q_values": Q_VALUES,
        "direct_fit": direct_fit,
        "q_eff_vs_L_corr_per_block": q_l_corr,
    }
    (RUNS_DIR / "pq_signature_summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote runs/pq_signature_summary.json")

    # ===== Plots =====
    b_arr = np.arange(N_BLOCKS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.errorbar(b_arr, per_block["p_eff_mean"], yerr=per_block["p_eff_std"],
                 fmt="-o", color="C2", markersize=3, capsize=3, label="p_eff (positive signature)")
    ax.errorbar(b_arr, per_block["q_eff_mean"], yerr=per_block["q_eff_std"],
                 fmt="-s", color="C3", markersize=3, capsize=3, label="q_eff (NEGATIVE signature)")
    ax.axhline(3, color="gray", ls="--", alpha=0.5, label="rank-3 Euclidean (ground truth)")
    ax.set_ylabel("effective signature dimension (|λ| > 1% max)", fontsize=10)
    ax.set_title("Pseudo-Euclidean signature (p, q) of the trunk's decoded distance maps across depth\n"
                  "Generic indefinite signature with q ≫ 1 throughout — distinct from PSD-cone (q=0) or hyperbolic (q=1)",
                  fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ratio = np.array(per_block["q_eff_mean"]) / np.maximum(np.array(per_block["p_eff_mean"]), 1)
    ax.plot(b_arr, ratio, "-o", color="C4", markersize=4)
    ax.set_xlabel("trunk block", fontsize=10)
    ax.set_ylabel("q_eff / p_eff", fontsize=10)
    ax.set_title("Ratio of negative-signature to positive-signature dimensions\n"
                  "(M-shape mirrors triangle-violation; falls to zero at block 47)",
                  fontsize=11)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("runs/pq_signature_headline.png", dpi=130, bbox_inches="tight")
    log.info("saved runs/pq_signature_headline.png")

    # Direct fit plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for b in fit_blocks:
        errs = [direct_fit[b][q] for q in Q_VALUES]
        ax.plot(Q_VALUES, errs, "-o", markersize=5, label=f"block {b}")
    ax.set_xscale("symlog")
    ax.set_xlabel(f"# negative dimensions in pseudo-Euclidean fit (p fixed at {P_FIXED})", fontsize=10)
    ax.set_ylabel("relative reconstruction error of D²", fontsize=10)
    ax.set_title("Direct pseudo-Euclidean fit: how many negative dimensions do we need?\n"
                  "Lower curve = block's distance map needs more negative dims to embed",
                  fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("runs/pq_fit_quality.png", dpi=130, bbox_inches="tight")
    log.info("saved runs/pq_fit_quality.png")


if __name__ == "__main__":
    main()
