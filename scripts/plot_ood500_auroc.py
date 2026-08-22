#!/usr/bin/env python3
"""AUROC comparison on the OOD-500 dataset.

Metrics compared (all evaluated on TM-score > 0.5 binary classification):
  • pLDDT              – ESMFold's built-in confidence
  • min_last5          – minimum corr_vs_47 over last-5 layers (layers 42-46)
  • late_amp           – amplitude of convergence curve over blocks 35–46
  • corr_start         – corr_vs_47 at the earliest layer (layer 1 with cw30 probes)
  • pce_raw            – Kadane max-positive-curvature on raw second derivative
  • pce_smooth         – same but on Gaussian-smoothed curve (σ=1.5)

Output: 2-row figure (scatter + AUROC bar chart)
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

# ── paths ─────────────────────────────────────────────────────────────────────
OOD_DIR     = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_ood500_native_cw30_stream"
SUMMARY_CSV = OOD_DIR / "general_convergence_summary.csv"
OUT_PATH    = OOD_DIR / "ood500_auroc_comparison.png"


# ── style ─────────────────────────────────────────────────────────────────────
def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


# ── Kadane max positive curvature ─────────────────────────────────────────────
def kadane_max(arr: np.ndarray) -> float:
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def pce_smooth(corr: np.ndarray, sigma: float = 1.5) -> float:
    c_sm = gaussian_filter1d(corr.astype(float), sigma=sigma)
    d2 = np.diff(c_sm, n=2)
    return kadane_max(np.maximum(d2, 0.0))


def pce_raw(corr: np.ndarray) -> float:
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))


# ── AUROC ─────────────────────────────────────────────────────────────────────
def auroc(scores: np.ndarray, fail_labels: np.ndarray) -> float:
    """AUROC where higher score predicts failure."""
    n_pos = fail_labels.sum(); n_neg = len(fail_labels) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    thresholds = np.linspace(scores.min(), scores.max(), 600)
    fprs, tprs = [], []
    for t in thresholds:
        pred = scores >= t
        tprs.append((pred & fail_labels).sum() / n_pos)
        fprs.append((pred & ~fail_labels).sum() / n_neg)
    idx = np.argsort(fprs)
    return float(np.trapezoid(np.array(tprs)[idx], np.array(fprs)[idx]))


# ── data loading ──────────────────────────────────────────────────────────────
def load_data() -> list[dict]:
    summary = {r["protein"]: r for r in csv.DictReader(SUMMARY_CSV.open())}

    data = []
    for d in sorted(OOD_DIR.glob("*_analysis")):
        corr_p = d / "layerwise_distance_correlations.csv"
        if not corr_p.exists():
            continue
        name = d.name.removesuffix("_analysis")
        s = summary.get(name, {})
        if not s:
            continue
        tm = float(s.get("tm_score", "nan"))
        plddt = float(s.get("plddt", "nan"))
        if np.isnan(tm) or np.isnan(plddt):
            continue

        rows = list(csv.DictReader(corr_p.open()))
        layers = np.array([int(r["layer"]) for r in rows])
        corr_arr = np.array([float(r["corr_vs_47"]) for r in rows])

        # Use only layers 1-46 for derived metrics (layer 47 is reference itself)
        mask46 = layers <= 46
        layers46 = layers[mask46]
        corr46 = corr_arr[mask46]

        # late_amp: blocks 35–46
        idx35 = layers46 >= 35
        c_late = corr46[idx35] if idx35.sum() > 1 else corr46
        late_amp = float(c_late.max() - c_late.min())

        # min_last5: layers 42-46
        idx42 = layers46 >= 42
        min_last5 = float(corr46[idx42].min()) if idx42.sum() > 0 else float(corr46.min())

        data.append({
            "name":        name,
            "tm":          tm,
            "plddt":       plddt,
            "min_last5":   min_last5,
            "corr_start":  float(corr46[0]),
            "late_amp":    late_amp,
            "pce_raw":     pce_raw(corr46),
            "pce_smooth":  pce_smooth(corr46, sigma=1.5),
        })

    return data


# ── scatter panel ─────────────────────────────────────────────────────────────
FAIL_THRESH = 0.5

METRIC_DEFS = [
    # (col_key, display_label, x_label, failure_direction, color)
    ("plddt",      "pLDDT",              "ESMFold mean pLDDT",              "low",  "#2563eb"),
    ("min_last5",  "min_last5",          "Min corr_vs_47 (layers 42–46)",   "low",  "#7c3aed"),
    ("late_amp",   "late_amp",           "Late-layer amplitude (35–46)",    "high", "#ea580c"),
    ("pce_raw",    "pce_raw",            "Kadane max curvature (raw)",      "high", "#16a34a"),
    ("pce_smooth", "pce_smooth (σ=1.5)", "Kadane max curvature (smoothed)", "high", "#0891b2"),
    ("corr_start", "corr_start (L1)",    "Probe–distogram corr @ layer 1",  "low",  "#be185d"),
]


def scatter_with_auroc(ax, x, y, labels, auc_val, x_label, title, color):
    fail_mask = labels.astype(bool)
    ax.scatter(x[~fail_mask], y[~fail_mask],
               color="#94a3b8", s=50, alpha=0.35, edgecolors="none",
               label=f"success (n={(~fail_mask).sum()})", zorder=2)
    ax.scatter(x[fail_mask], y[fail_mask],
               color="#ef4444", s=80, alpha=0.85, edgecolors="#7f1d1d", lw=0.5,
               marker="^", label=f"failure TM<0.5 (n={fail_mask.sum()})", zorder=3)
    ax.axhline(FAIL_THRESH, color="#0f172a", lw=0.7, ls="--", alpha=0.4)

    auc_color = "#15803d" if auc_val >= 0.75 else (
        "#d97706" if auc_val >= 0.60 else "#dc2626")
    ax.text(0.04, 0.96, f"AUROC = {auc_val:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#a16207", alpha=0.95))
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("TM-score vs PDB", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, lw=0.5)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    style()
    data = load_data()
    n = len(data)
    print(f"n = {n} proteins")

    tm     = np.array([r["tm"]  for r in data])
    labels = (tm < FAIL_THRESH).astype(bool)
    print(f"failures={labels.sum()}, successes={(~labels).sum()}")

    aurocs = {}
    for key, label, _, fail_dir, _ in METRIC_DEFS:
        vals = np.array([r[key] for r in data])
        fail_score = -vals if fail_dir == "low" else vals
        aurocs[key] = auroc(fail_score, labels)

    print("\nAUROC summary:")
    for key, label, *_ in METRIC_DEFS:
        print(f"  {label:>28}: {aurocs[key]:.4f}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    ncols = len(METRIC_DEFS)
    fig = plt.figure(figsize=(ncols * 3.8, 13))
    gs  = fig.add_gridspec(2, ncols, left=0.04, right=0.99, top=0.89,
                            bottom=0.07, wspace=0.30, hspace=0.42)

    fig.suptitle(
        f"OOD-500: convergence metrics for fold-failure detection (AUROC on TM < 0.5)",
        x=0.05, y=0.96, ha="left", fontsize=17, fontweight="bold", color="#172033",
    )
    fig.text(
        0.05, 0.92,
        f"n={n} proteins (all synthetic designs, cw30 probes).  "
        f"{labels.sum()} failures (TM<0.5) / {(~labels).sum()} successes.  "
        "pce = Kadane max integral of positive second-derivative.  "
        "Smoothed pce: Gaussian σ=1.5 prevents noise from concave-down curves.",
        ha="left", fontsize=10, color="#475569",
    )

    # Row 0 – scatter plots
    for col, (key, label, x_label, fail_dir, color) in enumerate(METRIC_DEFS):
        ax = fig.add_subplot(gs[0, col])
        vals = np.array([r[key] for r in data])
        scatter_with_auroc(ax, vals, tm, labels, aurocs[key], x_label, label, color)

    # Row 1 – AUROC bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    keys_bar   = [k for k, *_ in METRIC_DEFS]
    labels_bar = [lbl for _, lbl, *_ in METRIC_DEFS]
    colors_bar = [c for *_, c in METRIC_DEFS]
    auc_vals   = [aurocs[k] for k in keys_bar]

    bars = ax_bar.barh(labels_bar, auc_vals, color=colors_bar,
                       edgecolor="#172033", lw=0.6, height=0.55)
    ax_bar.axvline(0.5, color="#94a3b8", lw=1.2, ls="--", alpha=0.7,
                   label="random baseline (0.50)")
    for bar, val in zip(bars, auc_vals):
        ax_bar.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=11, fontweight="bold",
                    color="#172033")
    ax_bar.set_xlim(0.0, 1.05)
    ax_bar.set_xlabel("AUROC (TM < 0.5 = failure)", fontsize=12)
    ax_bar.set_title("AUROC comparison — higher is better failure detector", fontsize=13)
    ax_bar.legend(fontsize=10, loc="lower right")
    ax_bar.grid(True, lw=0.5, axis="x")
    ax_bar.invert_yaxis()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
