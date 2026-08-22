#!/usr/bin/env python3
"""Threshold-sweep separability plot.

X-axis: the metric's threshold value (swept across its full range, normalised 0→1).
Y-axis: classification accuracy when using that threshold to predict success/failure.

Three overlaid curves: pLDDT, min_last5 (convergence), late_amp (sharpening).

Panel A — success defined by MAE < 2 Å
Panel B — success defined by RMSD < 5 Å  (proxy for TM-score > 0.5)

The key point: our convergence/sharpening metrics should show a WIDER PLATEAU
of high accuracy — pLDDT has heavy overlap so accuracy degrades quickly off-peak.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib as mpl

CSV_PATH = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native/general_convergence_n40_native_logmae_summary.csv"
RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#e2e8f0",
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def kabsch_align(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pc = p - p.mean(0); qc = q - q.mean(0)
    h = pc.T @ qc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1, 1, det]) @ u.T).T + q.mean(0)


def tm_score(pred_ca: np.ndarray, gt_ca: np.ndarray) -> float:
    n = min(len(pred_ca), len(gt_ca))
    if n < 3:
        return float("nan")
    p = pred_ca[:n]; g = gt_ca[:n]
    aligned = kabsch_align(p, g)
    dists = np.sqrt(((aligned - g) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))


def compute_tm(name: str, runs_root: Path) -> float:
    for pat in ("benchmark_{n}", "hard_synthetic_miss_{n}", "synthetic_design_{n}", "{n}"):
        d = runs_root / pat.format(n=name)
        pred_p = d / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_p   = d / "gt_ca_distances.npz"
        if pred_p.exists() and gt_p.exists():
            pred_ca = np.load(pred_p)["pred_ca"].astype(float)
            gt_ca   = np.load(gt_p)["ca_coords"].astype(float)
            return tm_score(pred_ca, gt_ca)
    return float("nan")


def load_data(csv_path: Path, runs_root: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        for k in ("mae", "rmsd", "plddt", "min_last5", "late_amp"):
            r[k] = float(r[k])
        r["tm"] = compute_tm(r["name"], runs_root)
    # Drop proteins without TM
    missing = [r["name"] for r in rows if np.isnan(r["tm"])]
    if missing:
        print(f"  no TM-score for: {missing} — excluded from TM panels")
    return rows


def threshold_accuracy(values: np.ndarray, labels: np.ndarray,
                        direction: str, n_pts: int = 300):
    """Sweep threshold; return (thresholds, accuracy, tpr, fpr) arrays.

    direction: 'high' → predict success if value >= threshold
               'low'  → predict success if value <= threshold
    """
    lo, hi = values.min(), values.max()
    thresholds = np.linspace(lo, hi, n_pts)
    acc, tpr, fpr = [], [], []
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    for t in thresholds:
        if direction == "high":
            pred = values >= t
        else:
            pred = values <= t
        tp = (pred & labels).sum()
        tn = (~pred & ~labels).sum()
        fp = (pred & ~labels).sum()
        fn = (~pred & labels).sum()
        acc.append((tp + tn) / len(labels))
        tpr.append(tp / n_pos if n_pos > 0 else 0)
        fpr.append(fp / n_neg if n_neg > 0 else 0)

    return thresholds, np.array(acc), np.array(tpr), np.array(fpr)


def best_accuracy(acc):
    return acc.max()


def panel(ax, rows, success_mask, metric_configs, title, n_pts=400, smooth_sigma=8):
    """Draw threshold-sweep accuracy curves for all metrics."""
    labels = np.array([success_mask(r) for r in rows])
    n_success = labels.sum()
    n_total = len(labels)

    # Chance level
    chance = max(n_success, n_total - n_success) / n_total
    ax.axhline(chance, color="#94a3b8", lw=1.4, ls="--", alpha=0.7,
               label=f"majority-class baseline ({chance:.2f})")

    palette = {
        "pLDDT":     ("#2563eb", "-",  2.5),
        "min_last5": ("#16a34a", "-",  2.5),
        "late_amp":  ("#dc2626", "-",  2.5),
    }

    best_accs = {}
    for label, vals_key, direction, x_label in metric_configs:
        vals = np.array([r[vals_key] for r in rows])
        color, ls, lw = palette[label]
        thresholds, acc, _, _ = threshold_accuracy(vals, labels, direction, n_pts)

        # Normalize x-axis to [0,1] so all metrics are directly comparable
        x_norm = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min() + 1e-12)

        # Smooth to reduce staircase noise from discrete n=40
        acc_smooth = gaussian_filter1d(acc.astype(float), sigma=smooth_sigma)

        best = best_accuracy(acc)
        best_accs[label] = best
        ax.plot(x_norm, acc_smooth, color=color, ls=ls, lw=lw,
                label=f"{label}  (peak acc={best:.2f})")
        # Faint raw
        ax.plot(x_norm, acc, color=color, lw=0.6, alpha=0.2)

    ax.set_xlabel("Metric threshold (normalised 0→1,  → = stricter)", fontsize=10)
    ax.set_ylabel("Classification accuracy", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlim(-0.01, 1.01)
    ax.legend(fontsize=10, loc="lower center")
    ax.grid(True, lw=0.6)

    # Annotate n
    ax.text(0.02, 0.03, f"n={n_total}  ({n_success} success / {n_total-n_success} failure)",
            transform=ax.transAxes, fontsize=9, color="#475569")

    return labels


def auroc(values, labels, direction):
    """Simple AUROC via trapezoid on sorted FPR/TPR."""
    lo, hi = values.min(), values.max()
    thresholds = np.linspace(lo, hi, 500)
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    fprs, tprs = [], []
    for t in thresholds:
        pred = (values >= t) if direction == "high" else (values <= t)
        tprs.append((pred & labels).sum() / n_pos)
        fprs.append((pred & ~labels).sum() / n_neg)
    fprs, tprs = np.array(fprs), np.array(tprs)
    idx = np.argsort(fprs)
    return float(np.trapz(tprs[idx], fprs[idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent /
                    "runs/general_convergence_n40_native/threshold_separability.png")
    args = ap.parse_args()

    style()
    rows = load_data(args.csv, RUNS_ROOT)
    print(f"Loaded {len(rows)} proteins")

    # metric_configs: (display_label, column_key, direction, short_label)
    # direction 'high' → higher metric = predict success
    # direction 'low'  → lower metric  = predict success
    metric_configs = [
        ("pLDDT",     "plddt",     "high", "high→success"),
        ("min_last5", "min_last5", "low",  "low→success"),
        ("late_amp",  "late_amp",  "high", "high→success"),
    ]

    # Two success definitions (TM panels filter rows without TM data)
    rows_tm = [r for r in rows if not np.isnan(r["tm"])]
    success_defs = [
        ("MAE < 2 Å  (distance accuracy)",  rows,    lambda r: r["mae"] < 2.0),
        ("TM-score > 0.5  (fold accuracy)", rows_tm, lambda r: r["tm"] > 0.5),
    ]

    # ── Combined 2×2 figure: threshold-sweep (top) + ROC (bottom) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.07, wspace=0.28, hspace=0.38)
    fig.suptitle(
        "Separability: convergence metrics draw a cleaner success/failure boundary than pLDDT",
        x=0.06, y=0.97, ha="left", fontsize=17, fontweight="bold", color="#172033"
    )
    fig.text(
        0.06, 0.93,
        "Top: accuracy at every threshold (normalised x-axis; faint = raw, solid = smoothed).  "
        "Bottom: ROC curves.  Wide high plateau / higher AUROC = cleaner separation.",
        ha="left", fontsize=11, color="#475569"
    )

    palette2 = {
        "pLDDT":     ("#2563eb", "-"),
        "min_last5": ("#16a34a", "-"),
        "late_amp":  ("#dc2626", "-"),
    }

    print("\nAUROC summary")
    print(f"{'metric':>12}  {'MAE<2 AUROC':>12}  {'TM>0.5 AUROC':>13}")

    aurocs: dict[str, dict] = {}
    for col, (def_label, row_subset, mask_fn) in enumerate(success_defs):
        labels = np.array([mask_fn(r) for r in row_subset])
        n_pos = labels.sum(); n_neg = len(labels) - n_pos

        # ── Row 0: threshold accuracy ─────────────────────────────────────
        panel(axes[0, col], row_subset, mask_fn, metric_configs, def_label, n_pts=400)

        # ── Row 1: ROC ───────────────────────────────────────────────────
        ax_roc = axes[1, col]
        ax_roc.plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, ls="--", label="random (0.50)")
        for label, vals_key, direction, _ in metric_configs:
            vals = np.array([r[vals_key] for r in row_subset])
            lo, hi = vals.min(), vals.max()
            thresholds = np.linspace(lo, hi, 600)
            fprs_l, tprs_l = [], []
            for t in thresholds:
                pred = (vals >= t) if direction == "high" else (vals <= t)
                fprs_l.append((pred & ~labels).sum() / n_neg)
                tprs_l.append((pred & labels).sum() / n_pos)
            fprs_arr = np.array(fprs_l); tprs_arr = np.array(tprs_l)
            idx = np.argsort(fprs_arr)
            auc_val = float(np.trapezoid(tprs_arr[idx], fprs_arr[idx]))
            aurocs.setdefault(label, {})[def_label] = auc_val
            color, ls = palette2[label]
            ax_roc.plot(fprs_arr[idx], tprs_arr[idx], color=color, ls=ls, lw=2.4,
                        label=f"{label}  AUROC={auc_val:.3f}")

        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.set_title(f"ROC — {def_label}", fontsize=12)
        ax_roc.legend(fontsize=10)
        ax_roc.grid(True, lw=0.6)
        ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.text(0.02, 0.03, f"n={len(row_subset)}  ({n_pos}+/{n_neg}−)",
                    transform=ax_roc.transAxes, fontsize=9, color="#475569")

    for label, _, _, _ in metric_configs:
        d = aurocs[label]
        vals_list = list(d.values())
        print(f"{label:>12}  {vals_list[0]:>12.3f}  {vals_list[1]:>13.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
