"""Analyze WHAT our K=2 physics bottleneck misses at each depth.

For a chosen set of blocks (default: 0 / 9 (dip) / 24 / 43 (peak) / 47 (final)),
load the trained K=2 PSA, compute the residual r = h - ĥ for every pair in the
val proteins, and run PCA on the residuals to characterize the *unexplained*
directions:

  * residual_fraction        = ||r||² / ||h||²
  * variance explained by top-{1, 5, 20} PCA components of r
  * the top-20 PCA components themselves (a small (20, 128) matrix per block)
  * cosine similarity of top components across blocks (do the same "non-physical"
    directions persist through depth, or change?)

If the non-physical residual is low-rank and stable across depth, that suggests
a fixed "extra channel" the model carries. If it's high-rank and unstable, the
unexplained content is structurally different at each layer.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from psa.models import PSA
from psa.targets_k2 import K_PAIR_K2, build_pair_targets_k2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("residuals")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
RUNS_DIR = ROOT / "runs"

BLOCKS_TO_ANALYZE = [0, 9, 24, 43, 47]
SEED = 0
TOP_PCA = 20


def split_by_protein(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_block_psa(block_idx: int) -> PSA:
    block_dir = RUNS_DIR / f"k2_block_{block_idx:02d}"
    model = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    model.load_state_dict(torch.load(block_dir / "model.pt",
                                       map_location="cpu", weights_only=True))
    model.eval()
    return model


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    _, val_ids = split_by_protein(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    per_block: dict[int, dict] = {}
    top_components: dict[int, np.ndarray] = {}

    for b in BLOCKS_TO_ANALYZE:
        log.info("Block %d: loading PSA, computing residuals…", b)
        psa = load_block_psa(b).to(device)

        residuals_all = []
        h_all = []
        for pdb_id in val_ids:
            act = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt",
                              map_location="cpu", weights_only=False)
            z = act["z"][b, 0].float()           # (L, L, 128)
            L = z.shape[0]
            _, keep = build_pair_targets_k2(ROOT / "data" / "physics"
                                              / f"{pdb_id.lower()}.npz")
            if int(keep.numel() ** 0.5) != L:
                continue
            x = z.reshape(L * L, 128)[keep].to(device)
            with torch.no_grad():
                _, h_hat = psa(x)
            r = (x - h_hat).cpu().numpy()
            residuals_all.append(r)
            h_all.append(x.cpu().numpy())

        R = np.concatenate(residuals_all, axis=0)        # (N_total, 128)
        H = np.concatenate(h_all, axis=0)
        log.info("  residuals shape %s", R.shape)

        # Residual fraction (matches 1 - recon_R²)
        ss_res = (R * R).sum()
        ss_tot = ((H - H.mean(0, keepdims=True)) ** 2).sum()
        resid_frac = float(ss_res / (ss_tot + 1e-8))

        # PCA on centered residuals via SVD on a subsample (RAM-safe)
        n = R.shape[0]
        sample = R if n <= 100_000 else R[np.random.choice(n, 100_000, replace=False)]
        sample_c = sample - sample.mean(0, keepdims=True)
        # SVD
        t0 = time.time()
        u, s, vt = np.linalg.svd(sample_c, full_matrices=False)
        log.info("  SVD in %.1fs (n=%d)", time.time() - t0, sample_c.shape[0])
        var_total = (s * s).sum()
        var_top1 = float((s[0] ** 2) / var_total)
        var_top5 = float((s[:5] ** 2).sum() / var_total)
        var_top20 = float((s[:20] ** 2).sum() / var_total)

        top_components[b] = vt[:TOP_PCA].copy()           # (20, 128)
        per_block[b] = {
            "residual_fraction": resid_frac,
            "1 - r2_recon (eval)": resid_frac,
            "var_explained_top1": var_top1,
            "var_explained_top5": var_top5,
            "var_explained_top20": var_top20,
            "n_pairs": int(R.shape[0]),
            "n_proteins": len(residuals_all),
        }
        log.info("  resid_frac=%.3f  top1=%.3f  top5=%.3f  top20=%.3f",
                  resid_frac, var_top1, var_top5, var_top20)

    # Cross-block component similarity: cosine of top-1 PC across all pairs of blocks
    log.info("\nTop-PC cosine similarity matrix across blocks (top-1):")
    blocks = BLOCKS_TO_ANALYZE
    sim = np.zeros((len(blocks), len(blocks)))
    for i, b1 in enumerate(blocks):
        for j, b2 in enumerate(blocks):
            v1 = top_components[b1][0]
            v2 = top_components[b2][0]
            sim[i, j] = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    header = "        " + "  ".join(f"{b:>5}" for b in blocks)
    log.info(header)
    for i, b in enumerate(blocks):
        log.info("  %3d  %s", b, "  ".join(f"{sim[i, j]:>5.2f}" for j in range(len(blocks))))

    out = {
        "blocks": BLOCKS_TO_ANALYZE,
        "per_block": per_block,
        "top_components": {str(b): top_components[b].tolist() for b in top_components},
        "top1_cosine_matrix": sim.tolist(),
    }
    out_path = RUNS_DIR / "residuals_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
