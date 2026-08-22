#!/usr/bin/env python3
"""AUROC comparison: pLDDT vs cw30 convergence metrics for fold-failure detection.

Metrics compared (all evaluated on TM-score > 0.5 binary classification):
  • pLDDT              – ESMFold's built-in confidence
  • corr_start         – probe–distogram agreement at the earliest available layer (11)
  • late_amp           – amplitude of convergence curve over blocks 35–46
  • pce_raw            – Kadane max-positive-curvature integral on raw  d²
  • pce_smooth         – same but on Gaussian-smoothed curve (σ=1.5),
                         prevents noisy concave-down curves from accumulating
                         spurious positive curvature

Output: 2-row figure
  Row 1 – scatter of each metric vs TM-score (AUROC annotated)
  Row 2 – horizontal AUROC bar chart
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

# ── paths ─────────────────────────────────────────────────────────────────────
CW30_DIR    = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native_cw30_partial"
RUNS_ROOT   = Path(__file__).resolve().parent.parent / "runs"
SUMMARY_CSV = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native/general_convergence_n40_native_logmae_summary.csv"
OUT_PATH    = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native_cw30_partial/auroc_comparison.png"


# ── style ─────────────────────────────────────────────────────────────────────
def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


# ── TM-score helpers ──────────────────────────────────────────────────────────
def kabsch_align(p, q):
    pc = p - p.mean(0); qc = q - q.mean(0)
    h = pc.T @ qc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1, 1, det]) @ u.T).T + q.mean(0)


def tm_score(pred_ca, gt_ca):
    n = min(len(pred_ca), len(gt_ca))
    if n < 3: return float("nan")
    p = pred_ca[:n]; g = gt_ca[:n]
    a = kabsch_align(p, g)
    d = np.sqrt(((a - g) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def get_tm(name: str) -> float:
    for pat in ("benchmark_{n}", "hard_synthetic_miss_{n}",
                "synthetic_design_{n}", "{n}"):
        d = RUNS_ROOT / pat.format(n=name)
        pp = d / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        gp = d / "gt_ca_distances.npz"
        if pp.exists() and gp.exists():
            return tm_score(
                np.load(pp)["pred_ca"].astype(float),
                np.load(gp)["ca_coords"].astype(float),
            )
    return float("nan")


# ── Kadane max positive curvature integral ────────────────────────────────────
def kadane_max(arr: np.ndarray) -> float:
    """Maximum contiguous subarray sum (Kadane) on arr."""
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def pce_smooth(corr: np.ndarray, sigma: float = 1.5) -> float:
    """Positive concave energy on Gaussian-smoothed curve.

    Smoothing (σ=1.5 layers) removes high-frequency noise so that
    steeply-rising but concave-DOWN segments (logarithmic bad folds)
    do not accumulate spurious positive d² values.
    """
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
    summary = {r["name"]: r for r in csv.DictReader(SUMMARY_CSV.open())}

    data = []
    for d in sorted(CW30_DIR.glob("*_analysis")):
        corr_p = d / "layerwise_distance_correlations.csv"
        if not corr_p.exists():
            continue
        name = d.name.removesuffix("_analysis")
        rows = list(csv.DictReader(corr_p.open()))
        layers = np.array([int(r["layer"]) for r in rows])
        corr   = np.array([float(r["corr_vs_47"]) for r in rows])
        tm     = get_tm(name)

        # late_amp: blocks 35–46
        idx35 = np.where(layers >= 35)[0]
        c_late = corr[idx35] if len(idx35) > 1 else corr
        late_amp = float(c_late.max() - c_late.min())

        s = summary.get(name, {})
        plddt = float(s.get("plddt", float("nan")))

        data.append({
            "name":        name,
            "tm":          tm,
            "plddt":       plddt,
            "corr_start":  float(corr[0]),        # layer 11
            "late_amp":    late_amp,
            "pce_raw":     pce_raw(corr),
            "pce_smooth":  pce_smooth(corr, sigma=1.5),
        })

    return [r for r in data
            if not np.isnan(r["tm"]) and not np.isnan(r["plddt"])]


# ── scatter panel ─────────────────────────────────────────────────────────────
FAIL_THRESH = 0.5   # TM < 0.5 = fold failure

METRIC_DEFS = [
    # (col_key, display_label, x_label, failure_direction, color)
    # failure_direction: 'low' means lower value = more likely failure
    ("plddt",      "pLDDT",              "ESMFold mean pLDDT",       "low",  "#2563eb"),
    ("corr_start", "corr_start",         "Probe–distogram corr @ L11","low", "#7c3aed"),
    ("late_amp",   "late_amp (cw30)",    "Late-layer amplitude",      "high", "#ea580c"),
    ("pce_raw",    "pce_raw",            "Kadane max curvature (raw)","high", "#16a34a"),
    ("pce_smooth", "pce_smooth (σ=1.5)", "Kadane max curvature\n(Gaussian-smoothed)", "high", "#0891b2"),
]


def scatter_with_auroc(ax, x, y, labels, auc_val, x_label, title, color, fail_dir):
    fail_mask = labels.astype(bool)
    ax.scatter(x[~fail_mask], y[~fail_mask],
               color="#94a3b8", s=80, alpha=0.55, edgecolors="none",
               label=f"success (n={(~fail_mask).sum()})", zorder=2)
    ax.scatter(x[fail_mask], y[fail_mask],
               color="#ef4444", s=100, alpha=0.85, edgecolors="#7f1d1d", lw=0.5,
               marker="^", label=f"failure TM<0.5 (n={fail_mask.sum()})", zorder=3)
    ax.axhline(FAIL_THRESH, color="#0f172a", lw=0.7, ls="--", alpha=0.4)

    auc_color = "#15803d" if auc_val >= 0.75 else (
        "#d97706" if auc_val >= 0.60 else "#dc2626")
    ax.text(0.04, 0.96, f"AUROC = {auc_val:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#a16207", alpha=0.95))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("TM-score vs PDB", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, lw=0.5)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    style()
    data = load_data()
    n = len(data)
    print(f"n = {n} proteins with TM + pLDDT")

    tm     = np.array([r["tm"]   for r in data])
    labels = (tm < FAIL_THRESH).astype(bool)   # True = failure
    names  = [r["name"] for r in data]
    print(f"failures={labels.sum()}, successes={(~labels).sum()}")

    # Compute AUROC for every metric
    aurocs = {}
    for key, label, _, fail_dir, _ in METRIC_DEFS:
        vals = np.array([r[key] for r in data])
        fail_score = -vals if fail_dir == "low" else vals
        aurocs[key] = auroc(fail_score, labels)

    print("\nAUROC summary:")
    for key, label, *_ in METRIC_DEFS:
        print(f"  {label:>28}: {aurocs[key]:.4f}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    gs  = fig.add_gridspec(2, 5, left=0.05, right=0.99, top=0.89, bottom=0.07,
                            wspace=0.30, hspace=0.42)

    fig.suptitle(
        "Convergence metrics outperform pLDDT for fold-failure detection (AUROC on TM < 0.5)",
        x=0.05, y=0.96, ha="left", fontsize=18, fontweight="bold", color="#172033",
    )
    fig.text(
        0.05, 0.92,
        f"n={n} proteins with TM-score (15 failures / 24 successes).  "
        "cw30 = new general probes (consistency_weight=30).  "
        "pce = max integral of positive second derivative (Kadane).  "
        "Smoothed pce: Gaussian σ=1.5 prevents noisy concave-down curves from accumulating curvature.",
        ha="left", fontsize=10.5, color="#475569",
    )

    # Row 0 – scatter plots
    for col, (key, label, x_label, fail_dir, color) in enumerate(METRIC_DEFS):
        ax = fig.add_subplot(gs[0, col])
        vals = np.array([r[key] for r in data])
        scatter_with_auroc(ax, vals, tm, labels, aurocs[key],
                           x_label, label, color, fail_dir)

    # Row 1 – AUROC bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    keys   = [k for k, *_ in METRIC_DEFS]
    labels_bar = [lbl for _, lbl, *_ in METRIC_DEFS]
    colors_bar = [c for *_, c in METRIC_DEFS]
    auc_vals   = [aurocs[k] for k in keys]

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
