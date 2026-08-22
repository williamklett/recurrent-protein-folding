#!/usr/bin/env python3
"""Error-detection separability plot.

Focus: of the proteins that actually fail (TM < 0.5 or MAE > 2Å),
       what fraction does each metric correctly flag at each threshold?

This is True Negative Rate (TNR / specificity) — the "easy successes"
don't dilute the picture any more. A wide, high plateau = the metric
reliably catches failures across a broad range of threshold choices.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib as mpl

CSV_PATH = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native/general_convergence_n40_native_logmae_summary.csv"
RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
OUT_PATH  = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native/error_detection_separability.png"


# ── helpers ──────────────────────────────────────────────────────────────────

def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def kabsch_align(p, q):
    pc = p - p.mean(0); qc = q - q.mean(0)
    h = pc.T @ qc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1, 1, det]) @ u.T).T + q.mean(0)


def tm_score(pred_ca, gt_ca):
    n = min(len(pred_ca), len(gt_ca))
    if n < 3: return float("nan")
    p = pred_ca[:n]; g = gt_ca[:n]
    aligned = kabsch_align(p, g)
    dists = np.sqrt(((aligned - g) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))


def get_tm(name, runs_root):
    for pat in ("benchmark_{n}", "hard_synthetic_miss_{n}", "synthetic_design_{n}", "{n}"):
        d = runs_root / pat.format(n=name)
        pred_p = d / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_p   = d / "gt_ca_distances.npz"
        if pred_p.exists() and gt_p.exists():
            return tm_score(
                np.load(pred_p)["pred_ca"].astype(float),
                np.load(gt_p)["ca_coords"].astype(float),
            )
    return float("nan")


def load_data(csv_path, runs_root):
    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        for k in ("mae", "rmsd", "plddt", "min_last5", "late_amp"):
            r[k] = float(r[k])
        r["tm"] = get_tm(r["name"], runs_root)
    return rows


# ── core sweep ───────────────────────────────────────────────────────────────

def tnr_sweep(metric_vals: np.ndarray, is_failure: np.ndarray,
              direction: str, n_pts: int = 400):
    """At each threshold, compute TNR = TN / (TN + FP) over actual failures only.

    direction 'high' → predict success if metric >= threshold (flag failure if metric < threshold)
    direction 'low'  → predict success if metric <= threshold (flag failure if metric > threshold)
    """
    lo, hi = metric_vals.min(), metric_vals.max()
    thresholds = np.linspace(lo, hi, n_pts)
    tnr = []
    n_fail = is_failure.sum()
    for t in thresholds:
        if direction == "high":
            pred_fail = metric_vals < t   # below threshold → predicted failure
        else:
            pred_fail = metric_vals > t   # above threshold → predicted failure
        tn = (pred_fail & is_failure).sum()
        tnr.append(tn / n_fail if n_fail > 0 else 0.0)
    return thresholds, np.array(tnr)


# ── panel ────────────────────────────────────────────────────────────────────

PALETTE = {
    "pLDDT":     ("#2563eb", 2.6),
    "min_last5": ("#16a34a", 2.6),
    "late_amp":  ("#dc2626", 2.6),
}

METRIC_CONFIGS = [
    # (display_label, column_key, direction)
    ("pLDDT",     "plddt",     "high"),
    ("min_last5", "min_last5", "low"),
    ("late_amp",  "late_amp",  "high"),
]


def draw_panel(ax, rows, is_failure_fn, title, n_pts=400, smooth=10):
    vals_all = {k: np.array([r[k] for r in rows]) for _, k, _ in METRIC_CONFIGS}
    is_fail  = np.array([is_failure_fn(r) for r in rows])
    n_fail   = is_fail.sum()
    n_total  = len(rows)

    print(f"\n{title}")
    print(f"  failures: {n_fail}/{n_total}")

    best_tnrs = {}
    for label, key, direction in METRIC_CONFIGS:
        vals = vals_all[key]
        thresholds, tnr = tnr_sweep(vals, is_fail, direction, n_pts)

        x_norm = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min() + 1e-12)
        tnr_smooth = gaussian_filter1d(tnr.astype(float), sigma=smooth)

        peak = tnr.max()
        best_tnrs[label] = peak
        color, lw = PALETTE[label]
        ax.plot(x_norm, tnr_smooth, color=color, lw=lw, label=f"{label}  (peak {peak:.0%})")
        ax.plot(x_norm, tnr, color=color, lw=0.5, alpha=0.18)

        print(f"  {label:>12}: peak TNR={peak:.3f}")

    ax.set_xlabel("Metric threshold (normalised 0 → 1,  → = stricter)", fontsize=10)
    ax.set_ylabel("Fraction of failures correctly flagged (TNR)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.01, 1.01)
    ax.legend(fontsize=11, loc="lower center")
    ax.grid(True, lw=0.6)
    ax.text(0.02, 0.04,
            f"n={n_total} total  |  {n_fail} actual failures",
            transform=ax.transAxes, fontsize=9, color="#475569")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    style()
    rows = load_data(CSV_PATH, RUNS_ROOT)
    rows_tm = [r for r in rows if not np.isnan(r["tm"])]
    print(f"Loaded {len(rows)} proteins ({len(rows_tm)} with TM-score)")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.12, wspace=0.28)

    fig.suptitle(
        "Error detection: how reliably does each metric flag actual failures?",
        x=0.06, y=0.97, ha="left", fontsize=17, fontweight="bold", color="#172033",
    )
    fig.text(
        0.06, 0.91,
        "Y-axis = fraction of true failures correctly caught at each threshold  "
        "(TNR / specificity). Easy successes excluded — only failures count here.  "
        "Wide, high curve = metric reliably catches bad folds.",
        ha="left", fontsize=11, color="#475569",
    )

    draw_panel(axes[0], rows,    lambda r: r["mae"] > 2.0, "MAE > 2 Å  (distance failure)")
    draw_panel(axes[1], rows_tm, lambda r: r["tm"]  < 0.5, "TM-score < 0.5  (fold failure)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
