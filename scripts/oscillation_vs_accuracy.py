#!/usr/bin/env python3
"""
Multi-protein test: does the oscillation/convergence signature predict accuracy?

For each protein with both
  (a) a layerwise MDS structure trajectory  (distogram_mds_models.pdb)
  (b) a ground-truth crystal structure       (data/*.pdb)

we compute:
  • oscillation metrics       — tortuosity, lag-1 step autocorr, osc_score
  • accuracy of final layer   — Cα RMSD (Kabsch), Cα distance MAE, distance corr
                                 vs the ground-truth crystal

Then we look at the relationship across proteins.

Hypothesis: protein where the trunk oscillated (orbit signature) → accurate;
            where it drifted (open trajectory) → inaccurate.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent

# (label, layerwise_mds_pdb, ground_truth_pdb, ground_truth_chain)
PROTEINS = [
    ("1AO6 (albumin)",
     "runs/remote/esmfold_probe_1ao6_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
     "data/1AO6.pdb", "A"),
    ("T1137s6",
     "runs/remote/esmfold_probe_T1137s6_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
     "data/casp15/T1137/T1137s6.pdb", " "),
    ("8FMI (KRas)",
     "runs/remote/esmfold_probe_8fmi_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
     "data/8FMI.pdb", "A"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_layerwise_pdb(path: Path) -> np.ndarray:
    """Return Cα coords (n_models, n_residues, 3) from a multi-MODEL PDB."""
    models, current = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ENDMDL"):
                models.append(current)
                current = None
            elif line.startswith("ATOM") and current is not None:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                current.append([x, y, z])
    return np.array(models)


def parse_groundtruth_ca(path: Path, chain: str) -> np.ndarray:
    """Return (R, 3) Cα coords from chain `chain` of a single-structure PDB."""
    coords = []
    seen_resseq = set()
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            res_seq = int(line[22:26])
            if res_seq in seen_resseq:   # skip altloc dupes
                continue
            seen_resseq.add(res_seq)
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords.append([x, y, z])
    return np.array(coords)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q. Both (R, 3)."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return Pc @ R.T + Q.mean(0)


def ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def align_trajectory(coords: np.ndarray) -> np.ndarray:
    ref = coords[0]
    return np.stack([ref] + [kabsch_align(c, ref) for c in coords[1:]])


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """(R, 3) → (R, R) distance matrix"""
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def trajectory_metrics(coords_aligned: np.ndarray) -> dict:
    L = coords_aligned.shape[0]
    centroid = coords_aligned.mean(axis=0)
    d_to_centroid = np.array([ca_rmsd(c, centroid) for c in coords_aligned])
    steps = np.array([ca_rmsd(coords_aligned[t+1], coords_aligned[t])
                      for t in range(L-1)])
    end_to_end = ca_rmsd(coords_aligned[0], coords_aligned[-1])

    # Step autocorrelation (lag 1)
    flat = coords_aligned.reshape(L, -1)
    deltas = flat[1:] - flat[:-1]
    deltas /= np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8
    lag1 = float((deltas[:-1] * deltas[1:]).sum(axis=1).mean())

    # First negative-correlation lag
    first_neg = None
    for k in range(1, min(20, deltas.shape[0])):
        ips = (deltas[:-k] * deltas[k:]).sum(axis=1)
        if ips.mean() < 0:
            first_neg = k
            break

    return {
        "tortuosity":       steps.sum() / max(end_to_end, 1e-8),
        "path_length":      float(steps.sum()),
        "end_to_end":       float(end_to_end),
        "osc_score":        float(d_to_centroid.std() / max(d_to_centroid.mean(), 1e-8)),
        "mean_d_centroid":  float(d_to_centroid.mean()),
        "lag1_autocorr":    lag1,
        "first_neg_lag":    first_neg if first_neg is not None else 99,
        "mean_step":        float(steps.mean()),
    }


def accuracy_metrics(layer47: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compare model's layer-47 prediction to ground truth.
    Truncate to common length (N-terminal alignment).
    """
    n = min(layer47.shape[0], gt.shape[0])
    pred = layer47[:n]
    truth = gt[:n]

    # Kabsch-aligned RMSD
    pred_aligned = kabsch_align(pred, truth)
    rmsd = ca_rmsd(pred_aligned, truth)

    # Distance-matrix metrics (rotation-invariant)
    Dp = pairwise_distances(pred)
    Dt = pairwise_distances(truth)
    iu = np.triu_indices(n, k=1)
    mae = float(np.abs(Dp[iu] - Dt[iu]).mean())
    corr = float(np.corrcoef(Dp[iu], Dt[iu])[0, 1])

    return {
        "n_compared":   n,
        "ca_rmsd":      rmsd,
        "distance_mae": mae,
        "distance_corr": corr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    rows: list[dict] = []

    for label, lw_pdb, gt_pdb, gt_chain in PROTEINS:
        lw_path = ROOT / lw_pdb
        gt_path = ROOT / gt_pdb
        if not lw_path.exists() or not gt_path.exists():
            print(f"[skip] {label}: missing files")
            continue

        print(f"\n── {label}")
        coords = parse_layerwise_pdb(lw_path)             # (47, R_mod, 3)
        coords_aligned = align_trajectory(coords)
        traj = trajectory_metrics(coords_aligned)

        gt = parse_groundtruth_ca(gt_path, gt_chain)      # (R_gt, 3)
        acc = accuracy_metrics(coords[-1], gt)            # layer 47 vs GT

        # Print
        print(f"  shape          : {coords.shape}  (mod), GT={gt.shape}")
        print(f"  tortuosity     : {traj['tortuosity']:.3f}")
        print(f"  lag-1 autocorr : {traj['lag1_autocorr']:+.3f}")
        print(f"  osc_score      : {traj['osc_score']:.3f}")
        print(f"  first_neg_lag  : {traj['first_neg_lag']}")
        print(f"  ── accuracy of layer 47 vs ground truth ({acc['n_compared']} residues)")
        print(f"  Cα RMSD        : {acc['ca_rmsd']:.2f} Å  (Kabsch-aligned)")
        print(f"  distance MAE   : {acc['distance_mae']:.2f} Å")
        print(f"  distance corr  : {acc['distance_corr']:.3f}")

        rows.append({"label": label, **traj, **acc})

    # ──────────────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("SUMMARY: oscillation vs accuracy")
    print("=" * 92)
    print(f"{'protein':>18}  {'tortuosity':>10}  {'lag1':>7}  "
          f"{'first<0':>8}  {'osc_sc':>7}  {'RMSD(A)':>9}  {'dist MAE':>9}  {'dist r':>7}")
    for r in rows:
        print(f"{r['label']:>18}  {r['tortuosity']:10.3f}  "
              f"{r['lag1_autocorr']:+7.3f}  {r['first_neg_lag']:>8d}  "
              f"{r['osc_score']:7.3f}  {r['ca_rmsd']:9.2f}  "
              f"{r['distance_mae']:9.2f}  {r['distance_corr']:7.3f}")

    if len(rows) < 2:
        print("\nNot enough proteins to correlate.")
        return

    # ──────────────────────────────────────────────────────────────────────
    # Correlations across proteins
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("CROSS-PROTEIN CORRELATIONS (more positive = better fit predicts more oscillation)")
    print("=" * 92)

    metrics_for_corr = ["tortuosity", "lag1_autocorr", "first_neg_lag", "osc_score"]
    accuracies       = ["ca_rmsd", "distance_mae", "distance_corr"]

    # Note: low RMSD/MAE = high accuracy, high distance_corr = high accuracy.
    # We'll print correlations of each oscillation metric with each accuracy metric,
    # and note the EXPECTED sign for the "oscillation predicts correctness" claim.

    expected_signs = {
        ("tortuosity",    "ca_rmsd"):       "negative",  # more orbit → lower RMSD
        ("tortuosity",    "distance_mae"):  "negative",
        ("tortuosity",    "distance_corr"): "positive",
        ("lag1_autocorr", "ca_rmsd"):       "negative",  # more negative autocorr (orbit) → lower RMSD
        ("lag1_autocorr", "distance_mae"):  "negative",  # i.e., orbit (neg lag1) goes with low MAE = negative corr
        ("lag1_autocorr", "distance_corr"): "negative",  # orbit (neg lag1) goes with high corr = negative corr
        ("first_neg_lag", "ca_rmsd"):       "positive",  # late first negative (drift) → higher RMSD
        ("first_neg_lag", "distance_mae"):  "positive",
        ("first_neg_lag", "distance_corr"): "negative",
        ("osc_score",     "ca_rmsd"):       "positive",  # high osc_score (drift) → higher RMSD
        ("osc_score",     "distance_mae"):  "positive",
        ("osc_score",     "distance_corr"): "negative",
    }

    for m in metrics_for_corr:
        x = np.array([r[m] for r in rows], dtype=float)
        for a in accuracies:
            y = np.array([r[a] for r in rows], dtype=float)
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            exp = expected_signs.get((m, a), "?")
            actual = "positive" if r > 0 else "negative"
            agree = "✓" if exp == actual else "✗"
            print(f"  r({m:>15}, {a:>13}) = {r:+.3f}  (expected {exp:>8}, got {actual:>8})  {agree}")

    # ──────────────────────────────────────────────────────────────────────
    # Figure
    # ──────────────────────────────────────────────────────────────────────
    out_dir = ROOT / "runs" / "breathing_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_specs = [
        ("tortuosity",    "ca_rmsd",       axes[0, 0], "Tortuosity vs Cα RMSD"),
        ("lag1_autocorr", "ca_rmsd",       axes[0, 1], "Lag-1 autocorr vs Cα RMSD"),
        ("tortuosity",    "distance_corr", axes[1, 0], "Tortuosity vs distance correlation"),
        ("lag1_autocorr", "distance_corr", axes[1, 1], "Lag-1 autocorr vs distance correlation"),
    ]
    colors = ["#1976D2", "#D32F2F", "#388E3C", "#7B1FA2", "#F57C00"]
    for (xkey, ykey, ax, title) in plot_specs:
        for i, r in enumerate(rows):
            ax.scatter(r[xkey], r[ykey], s=140, color=colors[i % len(colors)],
                       edgecolors="black", linewidth=0.8, label=r["label"])
        ax.set_xlabel(xkey); ax.set_ylabel(ykey)
        ax.set_title(title); ax.grid(alpha=0.3)
        if len(rows) >= 2:
            x = np.array([r[xkey] for r in rows]); y = np.array([r[ykey] for r in rows])
            if np.std(x) > 0 and np.std(y) > 0:
                rr = np.corrcoef(x, y)[0, 1]
                ax.text(0.02, 0.97, f"r = {rr:+.3f}", transform=ax.transAxes,
                        va="top", ha="left", fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        if ax is axes[0, 0]:
            ax.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    p = out_dir / "fig4_oscillation_vs_accuracy.png"
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
