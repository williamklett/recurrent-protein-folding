"""Inspect what physics the trained PSAs recover at each trunk block.

For a chosen protein (default: a val-set protein not seen during PSA training),
load every per-block PSA, run PV on the protein's cached pair_rep, and
visualize how predicted physical maps evolve through depth.

Three figures are produced:

  1. contact_evolution.png  — predicted contact map at each block + ground truth
  2. distance_evolution.png — predicted distance map at each block + ground truth
  3. pair_trajectories.png  — for a few specific (i, j), plot all 5 z_phys
                              channels across block depth

Usage:
    python scripts/inspect_physics.py --pdb 1UBQ
    python scripts/inspect_physics.py --pdb 1CRN --blocks "0,5,9,15,24,35,43,47"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from psa.models import PSA
from psa.targets import (
    FeatureStats,
    K_PAIR,
    PAIR_FEATURE_NAMES,
    PAIR_FEATURE_SCHEMA,
    build_pair_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("inspect_physics")

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "runs" / "inspect"


def load_block_psa(block_idx: int) -> tuple[PSA, FeatureStats]:
    """Load the trained PSA + feature stats for one block."""
    block_dir = RUNS_DIR / f"block_{block_idx:02d}"
    model = PSA(d_pair=128, d_hidden=128, K=K_PAIR)
    model.load_state_dict(torch.load(block_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    stats_d = torch.load(block_dir / "stats.pt", map_location="cpu", weights_only=True)
    stats = FeatureStats(mean=stats_d["mean"], std=stats_d["std"],
                          is_binary=stats_d["is_binary"])
    return model, stats


def predict_all_blocks(pdb_id: str, num_blocks: int = 48) -> tuple[np.ndarray, dict, int]:
    """Return:
      preds : (num_blocks, L, L, K)   predictions in INTERPRETABLE units
                                       (denorm for regression, sigmoid for binary)
      truth : dict of ground-truth maps
      L     : sequence length
    """
    act = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
    z_all = act["z"].float()  # (48, 1, L, L, 128)
    L = z_all.shape[2]

    preds = np.zeros((num_blocks, L, L, K_PAIR), dtype=np.float32)
    for b in range(num_blocks):
        psa, stats = load_block_psa(b)
        with torch.no_grad():
            h = z_all[b, 0].reshape(L * L, 128)  # (L*L, 128)
            z_pred = psa.pv(h)                     # (L*L, K), raw outputs
        z_pred = z_pred.numpy().reshape(L, L, K_PAIR)

        # For each feature, convert to interpretable units.
        for k, (_, kind) in enumerate(PAIR_FEATURE_SCHEMA):
            if kind == "bin":
                # logit → probability
                preds[b, :, :, k] = 1.0 / (1.0 + np.exp(-z_pred[:, :, k]))
            else:
                # de-normalize: y_raw = y_norm * std + mean
                preds[b, :, :, k] = z_pred[:, :, k] * float(stats.std[k]) + float(stats.mean[k])

    # Ground truth
    y_flat, _ = build_pair_targets(PHYS_DIR / f"{pdb_id.lower()}.npz", mask_diagonal=False)
    y_maps = y_flat.numpy().reshape(L, L, K_PAIR)
    truth = {name: y_maps[:, :, k] for k, name in enumerate(PAIR_FEATURE_NAMES)}
    return preds, truth, L


def plot_map_evolution(
    preds: np.ndarray, truth: dict, feature_idx: int, feature_name: str,
    blocks_to_show: list[int], pdb_id: str, L: int,
    vmin: float | None = None, vmax: float | None = None, cmap: str = "viridis",
) -> Path:
    """Show predicted maps at selected blocks + ground truth, side by side."""
    n_panels = len(blocks_to_show) + 1
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.3 * nrows),
                              squeeze=False)
    axes = axes.flatten()

    truth_map = truth[feature_name]
    if vmin is None:
        vmin = min(truth_map.min(), preds[:, :, :, feature_idx].min())
    if vmax is None:
        vmax = max(truth_map.max(), preds[:, :, :, feature_idx].max())

    # Panel 0: ground truth
    im = axes[0].imshow(truth_map, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    axes[0].set_title(f"ground truth\n{feature_name}", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for i, b in enumerate(blocks_to_show, start=1):
        ax = axes[i]
        ax.imshow(preds[b, :, :, feature_idx], vmin=vmin, vmax=vmax, cmap=cmap,
                  origin="lower")
        ax.set_title(f"block {b}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(n_panels, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"{pdb_id}  L={L}  — {feature_name} evolution across depth",
                 fontsize=12, y=1.02)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(feature_name, fontsize=9)

    out = FIG_DIR / f"{pdb_id.lower()}_{feature_name}_evolution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pair_trajectories(
    preds: np.ndarray, truth: dict, pair_examples: list[tuple[int, int, str]],
    pdb_id: str, num_blocks: int = 48,
) -> Path:
    """For each (i, j) example, plot all 5 z_phys channels vs block depth.

    pair_examples : list of (i, j, label) tuples.
    """
    n = len(pair_examples)
    fig, axes = plt.subplots(K_PAIR, n, figsize=(3.5 * n, 2.2 * K_PAIR), sharex=True)
    if n == 1:
        axes = axes[:, None]
    if K_PAIR == 1:
        axes = axes[None, :]

    x = np.arange(num_blocks)
    for col, (i, j, label) in enumerate(pair_examples):
        for k, name in enumerate(PAIR_FEATURE_NAMES):
            ax = axes[k, col]
            ax.plot(x, preds[:, i, j, k], "-o", markersize=3, linewidth=1.3,
                    color="C0", label="predicted")
            truth_val = truth[name][i, j]
            ax.axhline(truth_val, color="C3", linestyle="--", linewidth=1.2,
                       label="ground truth")
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
            if k == 0:
                ax.set_title(f"pair ({i}, {j})  {label}", fontsize=10)
            if k == K_PAIR - 1:
                ax.set_xlabel("block", fontsize=10)
            if k == 0 and col == n - 1:
                ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"{pdb_id} — per-pair physics trajectory across blocks", fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / f"{pdb_id.lower()}_pair_trajectories.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def pick_interesting_pairs(truth: dict, L: int) -> list[tuple[int, int, str]]:
    """Heuristic: find one short-range contact, one long-range contact, one non-contact."""
    contacts = truth["contact"]
    seq_sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])

    # Short-range contact: contact with |i-j| in [4, 8], picking the most central one
    short_mask = contacts.astype(bool) & (seq_sep >= 4) & (seq_sep <= 8) & (np.tri(L, L, -1, dtype=bool).T)
    short_idx = np.argwhere(short_mask)
    short_pair = tuple(short_idx[len(short_idx) // 2]) if len(short_idx) else (0, 4)

    # Long-range contact: |i-j| > 15
    long_mask = contacts.astype(bool) & (seq_sep > 15) & (np.tri(L, L, -1, dtype=bool).T)
    long_idx = np.argwhere(long_mask)
    long_pair = tuple(long_idx[len(long_idx) // 2]) if len(long_idx) else (0, 20)

    # Non-contact: |i-j| > 15 and NOT contact, with d_ij in middle range
    non_mask = (~contacts.astype(bool)) & (seq_sep > 15) & (np.tri(L, L, -1, dtype=bool).T)
    non_idx = np.argwhere(non_mask)
    non_pair = tuple(non_idx[len(non_idx) // 2]) if len(non_idx) else (0, 25)

    return [
        (int(short_pair[0]), int(short_pair[1]), "short-range contact"),
        (int(long_pair[0]), int(long_pair[1]), "long-range contact"),
        (int(non_pair[0]), int(non_pair[1]), "non-contact"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", type=str, default="1UBQ", help="PDB id of a cached protein")
    ap.add_argument("--blocks", type=str, default="0,5,9,15,24,35,43,47",
                     help="comma-separated blocks to show in map evolution")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    blocks_to_show = [int(b) for b in args.blocks.split(",")]
    pdb_id = args.pdb.upper()

    log.info("Running PV across all 48 blocks for %s…", pdb_id)
    preds, truth, L = predict_all_blocks(pdb_id)
    log.info("  preds shape = %s   L = %d", preds.shape, L)

    log.info("Plotting contact-map evolution…")
    p = plot_map_evolution(preds, truth, feature_idx=PAIR_FEATURE_NAMES.index("contact"),
                            feature_name="contact", blocks_to_show=blocks_to_show,
                            pdb_id=pdb_id, L=L, vmin=0, vmax=1, cmap="Blues")
    log.info("  wrote %s", p)

    log.info("Plotting distance-map evolution…")
    p = plot_map_evolution(preds, truth, feature_idx=PAIR_FEATURE_NAMES.index("distance"),
                            feature_name="distance", blocks_to_show=blocks_to_show,
                            pdb_id=pdb_id, L=L, vmin=0, vmax=40, cmap="viridis_r")
    log.info("  wrote %s", p)

    log.info("Plotting hbond-map evolution…")
    p = plot_map_evolution(preds, truth, feature_idx=PAIR_FEATURE_NAMES.index("hbond"),
                            feature_name="hbond", blocks_to_show=blocks_to_show,
                            pdb_id=pdb_id, L=L, vmin=0, vmax=1, cmap="Purples")
    log.info("  wrote %s", p)

    log.info("Picking three example pairs and plotting trajectories…")
    pairs = pick_interesting_pairs(truth, L)
    for i, j, lab in pairs:
        log.info("  example pair (%d, %d) — %s (true distance=%.1f Å, contact=%d)",
                 i, j, lab, truth["distance"][i, j], int(truth["contact"][i, j]))
    p = plot_pair_trajectories(preds, truth, pairs, pdb_id)
    log.info("  wrote %s", p)


if __name__ == "__main__":
    main()
