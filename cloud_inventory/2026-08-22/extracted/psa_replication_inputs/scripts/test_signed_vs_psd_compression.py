"""Same-rank compression into signed vs Euclidean geometry.

For every decoded ESMFold block distance map D, form the centered Gram-like
matrix M = -1/2 J D^2 J. Then compress M with the same rank budget R under
different signature constraints:

  signed_abs_R:
      keep the R largest-|lambda| eigencomponents with their original signs.
      This is the best rank-R indefinite reconstruction in Frobenius norm.

  psd_pos_R:
      keep the R largest positive eigencomponents only. This is the natural
      rank-R Euclidean/PSD projection.

  psd_abs_flip_R:
      keep the same R largest-|lambda| eigenvectors as signed_abs_R but force
      all eigenvalues positive. This controls for using the same subspace/rank
      while removing the indefinite signature.

If the lift/project interpretation is right, signed_abs_R should beat the PSD
controls specifically in the middle indefinite phase, where q_eff is high.

Outputs:
  runs/signed_vs_psd_compression.json
  runs/signed_vs_psd_compression.png
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
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("signed_vs_psd")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
DISTO_DIR = ROOT / "data" / "distograms"
RUNS_DIR = ROOT / "runs"

N_BLOCKS = 48
SEED = 0
RANK_BUDGETS = [3, 8, 16, 32]


def split_proteins(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    s = pdb_ids.copy()
    rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def gram_from_dist(D: np.ndarray) -> np.ndarray:
    L = D.shape[0]
    J = np.eye(L) - np.ones((L, L)) / L
    M = -0.5 * J @ (D ** 2) @ J
    return 0.5 * (M + M.T)


def dist2_from_gram(M: np.ndarray) -> np.ndarray:
    diag = np.diag(M)
    return diag[:, None] + diag[None, :] - 2.0 * M


def decompose(M: np.ndarray):
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(np.abs(vals))[::-1]
    return vals, vecs, order


def reconstruct(vals: np.ndarray, vecs: np.ndarray, order: np.ndarray,
                rank: int, mode: str) -> np.ndarray:
    if mode == "signed_abs":
        keep = order[:rank]
        lam = vals[keep]
    elif mode == "psd_abs_flip":
        keep = order[:rank]
        lam = np.abs(vals[keep])
    elif mode == "psd_pos":
        pos = np.where(vals > 0)[0]
        pos = pos[np.argsort(vals[pos])[::-1]]
        keep = pos[:rank]
        lam = vals[keep]
    else:
        raise ValueError(mode)

    if len(keep) == 0:
        return np.zeros((vecs.shape[0], vecs.shape[0]), dtype=np.float64)
    U = vecs[:, keep]
    return (U * lam[None, :]) @ U.T


def rel_fro(target: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    diff = (target - recon)[mask]
    denom = np.linalg.norm(target[mask]) + 1e-9
    return float(np.linalg.norm(diff) / denom)


def effective_q(vals: np.ndarray, tau: float = 0.01) -> int:
    thr = tau * float(np.max(np.abs(vals)) + 1e-12)
    return int((vals < -thr).sum())


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows if (DISTO_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    proteins = []
    for pid in val_ids:
        d = torch.load(DISTO_DIR / f"{pid.lower()}.pt", map_location="cpu", weights_only=False)
        proteins.append({"pdb_id": pid, "L": int(d["L"]), "pred": d["pred_distance"].float().numpy()})
    log.info("Loaded %d distogram files", len(proteins))

    modes = ["signed_abs", "psd_pos", "psd_abs_flip"]
    per_item = []
    summary = {
        str(rank): {mode: [] for mode in modes} | {"q_eff": [], "signed_advantage_vs_psd_pos": [],
                                                    "signed_advantage_vs_psd_flip": []}
        for rank in RANK_BUDGETS
    }

    for b in range(N_BLOCKS):
        for rank in RANK_BUDGETS:
            block_errors = {mode: [] for mode in modes}
            block_q = []
            for p in proteins:
                D = p["pred"][b].astype(np.float64)
                M = gram_from_dist(D)
                vals, vecs, order = decompose(M)
                q = effective_q(vals)
                block_q.append(q)
                target_d2 = D ** 2
                mask = ~np.eye(D.shape[0], dtype=bool)
                row = {"pdb_id": p["pdb_id"], "L": p["L"], "block": b, "rank": rank, "q_eff": q}
                for mode in modes:
                    M_hat = reconstruct(vals, vecs, order, rank, mode)
                    d2_hat = dist2_from_gram(M_hat)
                    err = rel_fro(target_d2, d2_hat, mask)
                    block_errors[mode].append(err)
                    row[f"{mode}_rel_d2_error"] = err
                per_item.append(row)

            k = str(rank)
            for mode in modes:
                summary[k][mode].append(float(np.mean(block_errors[mode])))
            summary[k]["q_eff"].append(float(np.mean(block_q)))
            psd_pos = np.array(block_errors["psd_pos"])
            psd_flip = np.array(block_errors["psd_abs_flip"])
            signed = np.array(block_errors["signed_abs"])
            summary[k]["signed_advantage_vs_psd_pos"].append(float(np.mean(psd_pos - signed)))
            summary[k]["signed_advantage_vs_psd_flip"].append(float(np.mean(psd_flip - signed)))

    # Correlate signed advantage with q_eff across blocks.
    correlations = {}
    for rank in RANK_BUDGETS:
        k = str(rank)
        q = np.array(summary[k]["q_eff"])
        correlations[k] = {}
        for name in ["signed_advantage_vs_psd_pos", "signed_advantage_vs_psd_flip"]:
            adv = np.array(summary[k][name])
            r, p = pearsonr(q, adv)
            correlations[k][name] = {"pearson_r": float(r), "p": float(p)}

    out = {
        "rank_budgets": RANK_BUDGETS,
        "modes": modes,
        "blocks": list(range(N_BLOCKS)),
        "val_ids": val_ids,
        "summary": summary,
        "correlations_with_q_eff": correlations,
        "per_item": per_item,
    }
    out_path = RUNS_DIR / "signed_vs_psd_compression.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    for rank in RANK_BUDGETS:
        k = str(rank)
        log.info("\nRank %d:", rank)
        for b in [5, 9, 15, 20, 25, 30, 35, 40, 43, 47]:
            log.info(
                "  b%02d q=%.1f signed=%.3f psd_pos=%.3f psd_flip=%.3f adv_flip=%.3f",
                b,
                summary[k]["q_eff"][b],
                summary[k]["signed_abs"][b],
                summary[k]["psd_pos"][b],
                summary[k]["psd_abs_flip"][b],
                summary[k]["signed_advantage_vs_psd_flip"][b],
            )
        log.info("  corr(q_eff, signed advantage over psd_flip): r=%+.3f p=%.2g",
                 correlations[k]["signed_advantage_vs_psd_flip"]["pearson_r"],
                 correlations[k]["signed_advantage_vs_psd_flip"]["p"])

    blocks = np.arange(N_BLOCKS)
    fig, axes = plt.subplots(len(RANK_BUDGETS), 2, figsize=(13, 3.2 * len(RANK_BUDGETS)), sharex=True)
    for i, rank in enumerate(RANK_BUDGETS):
        k = str(rank)
        ax = axes[i, 0]
        ax.plot(blocks, summary[k]["signed_abs"], "-o", markersize=3, label="signed rank-R")
        ax.plot(blocks, summary[k]["psd_pos"], "-s", markersize=3, label="PSD positive rank-R")
        ax.plot(blocks, summary[k]["psd_abs_flip"], "-^", markersize=3, label="same subspace, signs flipped PSD")
        ax.set_ylabel(f"R={rank}\nrel D² error")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Same-rank reconstruction error")
            ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.plot(blocks, summary[k]["signed_advantage_vs_psd_flip"], "-o", color="C3", markersize=3,
                label="PSD flip error - signed error")
        ax2 = ax.twinx()
        ax2.plot(blocks, summary[k]["q_eff"], "-", color="0.35", alpha=0.55, label="q_eff")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel(f"R={rank}\nsigned advantage")
        ax2.set_ylabel("q_eff")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Does signed compression help where q_eff is high?")
    axes[-1, 0].set_xlabel("trunk block")
    axes[-1, 1].set_xlabel("trunk block")
    plt.tight_layout()
    fig_path = RUNS_DIR / "signed_vs_psd_compression.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    log.info("saved %s", fig_path)


if __name__ == "__main__":
    main()
