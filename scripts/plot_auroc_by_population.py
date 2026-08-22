#!/usr/bin/env python3
"""AUROC for fold-failure (TM < 0.5) detection across population slices.

Slices:
  all       — all 500 proteins (includes many easy successes)
  TM < 0.75 — mid + bad  (103 proteins)
  TM < 0.70 — mid + bad  (89 proteins, ~balanced 45/44)

Metrics: pLDDT, pce_raw, bt_raw, combined (z-score sum)

Panel layout:
  Row 0 — grouped AUROC bar chart per metric, coloured by population slice
  Row 1 — ROC curves for the TM < 0.70 slice (where the test is hardest)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT    = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "runs/general_convergence_ood500_native_cw30_stream"

FAIL_THRESH = 0.5
POP_CUTS    = [None, 0.75, 0.70]          # None = all
POP_LABELS  = ["All  (N=500)", "TM < 0.75", "TM < 0.70"]
POP_COLORS  = ["#94a3b8",      "#f59e0b",   "#ef4444"]


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


# ── metrics ───────────────────────────────────────────────────────────────────
def kadane_max(arr):
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def pce_raw_fn(corr):
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))


def bt_raw_fn(corr):
    res = np.clip(1.0 - corr, 1e-8, 1.0)
    return float(np.maximum(np.diff(res), 0.0).sum())


def zscore(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-12)


# ── data ──────────────────────────────────────────────────────────────────────
def load_data(ood_dir: Path) -> list[dict]:
    summary = {r["protein"]: r for r in csv.DictReader(
        (ood_dir / "general_convergence_summary.csv").open())}

    data = []
    for d in sorted(ood_dir.glob("*_analysis")):
        corr_p = d / "layerwise_distance_correlations.csv"
        if not corr_p.exists():
            continue
        name = d.name.removesuffix("_analysis")
        s = summary.get(name, {})
        if not s:
            continue
        tm    = float(s.get("tm_score", "nan"))
        plddt = float(s.get("plddt",    "nan"))
        if np.isnan(tm) or np.isnan(plddt):
            continue
        rows   = list(csv.DictReader(corr_p.open()))
        key    = "corr_vs_final_distogram" if "corr_vs_final_distogram" in rows[0] else "corr_vs_47"
        layers = np.array([int(r["layer"]) for r in rows])
        corr   = np.array([float(r[key])   for r in rows])
        corr46 = corr[layers <= 46]
        data.append({"name": name, "tm": tm, "plddt": plddt,
                      "pce_raw": pce_raw_fn(corr46),
                      "bt_raw":  bt_raw_fn(corr46)})

    # combined score calibrated on full dataset
    p = zscore(np.array([r["plddt"]   for r in data]))
    c = zscore(np.array([r["pce_raw"] for r in data]))
    b = zscore(np.array([r["bt_raw"]  for r in data]))
    combined = p - c - b
    for i, r in enumerate(data):
        r["combined"] = float(combined[i])
    return data


# ── AUROC ─────────────────────────────────────────────────────────────────────
def auroc(scores, fail_labels):
    n_pos = fail_labels.sum(); n_neg = len(fail_labels) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    thresholds = np.linspace(scores.min(), scores.max(), 800)
    fprs, tprs = [], []
    for t in thresholds:
        pred = scores >= t
        tprs.append((pred & fail_labels).sum() / n_pos)
        fprs.append((pred & ~fail_labels).sum() / n_neg)
    idx = np.argsort(fprs)
    return float(np.trapezoid(np.array(tprs)[idx], np.array(fprs)[idx]))


def roc_curve(scores, fail_labels, n=800):
    thresholds = np.linspace(scores.min(), scores.max(), n)
    n_pos = fail_labels.sum(); n_neg = len(fail_labels) - n_pos
    fprs, tprs = [], []
    for t in thresholds:
        pred = scores >= t
        tprs.append((pred & fail_labels).sum() / n_pos)
        fprs.append((pred & ~fail_labels).sum() / n_neg)
    fprs, tprs = np.array(fprs), np.array(tprs)
    idx = np.argsort(fprs)
    return fprs[idx], tprs[idx]


# ── metric definitions ────────────────────────────────────────────────────────
# (key, display label, failure direction: 'high' means higher score = failure)
METRICS = [
    ("plddt",    "pLDDT",    "low",  "#2563eb"),
    ("pce_raw",  "pce_raw",  "high", "#16a34a"),
    ("bt_raw",   "bt_raw",   "high", "#ea580c"),
    ("combined", "combined", "low",  "#7c3aed"),   # combined: lower = failure
]


def fail_score(data, key, direction):
    vals = np.array([r[key] for r in data])
    return -vals if direction == "low" else vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir", type=Path, default=OOD_DIR)
    ap.add_argument("--out",     type=Path, default=OOD_DIR / "auroc_by_population.png")
    args = ap.parse_args()

    style()
    data = load_data(args.ood_dir)
    print(f"loaded {len(data)} proteins")

    # Build population slices
    slices = []
    for cut, label, color in zip(POP_CUTS, POP_LABELS, POP_COLORS):
        subset = data if cut is None else [r for r in data if r["tm"] < cut]
        fail   = np.array([r["tm"] < FAIL_THRESH for r in subset])
        slices.append((subset, fail, label, color, cut))
        print(f"{label}: n={len(subset)}  failures={fail.sum()}  successes={(~fail).sum()}")

    # ── print AUROC table ─────────────────────────────────────────────────────
    print(f"\n{'metric':>10}", end="")
    for _, _, label, *_ in slices:
        print(f"  {label:>14}", end="")
    print()
    for key, mlabel, direction, _ in METRICS:
        print(f"{mlabel:>10}", end="")
        for subset, fail, *_ in slices:
            fs = fail_score(subset, key, direction)
            print(f"  {auroc(fs, fail):>14.4f}", end="")
        print()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.12,
                        wspace=0.32)
    fig.suptitle(
        "AUROC for fold-failure detection: does the metric still work on hard proteins?",
        x=0.06, y=0.96, ha="left", fontsize=17, fontweight="bold", color="#172033")
    fig.text(
        0.06, 0.905,
        f"Binary task: TM < {FAIL_THRESH} = failure.  "
        "Drug targets are often OOD — the TM < 0.70 slice (mid + bad, ~balanced) "
        "is the realistic operating regime.",
        ha="left", fontsize=11, color="#475569")

    # ── Panel 0: grouped bar chart ────────────────────────────────────────────
    ax = axes[0]
    n_metrics = len(METRICS)
    n_slices  = len(slices)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_slices - 1) / 2, (n_slices - 1) / 2, n_slices) * width

    for si, (subset, fail, slabel, scolor, _) in enumerate(slices):
        aucs = []
        for key, _, direction, _ in METRICS:
            fs = fail_score(subset, key, direction)
            aucs.append(auroc(fs, fail))
        bars = ax.bar(x + offsets[si], aucs, width * 0.88,
                      label=slabel, color=scolor, edgecolor="#172033", lw=0.6)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#172033")

    ax.axhline(0.5, color="#94a3b8", lw=1.2, ls="--", alpha=0.7, label="random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, *_ in METRICS], fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("AUROC by metric and population slice", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.5, axis="y")

    # ── Panel 1: ROC curves for TM < 0.70 slice (hardest test) ───────────────
    ax2 = axes[1]
    hard_subset, hard_fail, hard_label, _, _ = slices[-1]   # TM < 0.70
    ax2.plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, ls="--", label="random")
    for key, mlabel, direction, mcolor in METRICS:
        fs = fail_score(hard_subset, key, direction)
        fpr, tpr = roc_curve(fs, hard_fail)
        auc_val  = auroc(fs, hard_fail)
        ax2.plot(fpr, tpr, color=mcolor, lw=2.4,
                 label=f"{mlabel}  AUROC={auc_val:.3f}")
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title(f"ROC curves — {hard_label}\n(mid + bad proteins only)", fontsize=12)
    ax2.legend(fontsize=11, loc="lower right")
    ax2.grid(True, lw=0.5)
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
    n_sub = len(hard_subset)
    ax2.text(0.04, 0.06,
             f"n={n_sub}  ({hard_fail.sum()} failures / {(~hard_fail).sum()} mid)",
             transform=ax2.transAxes, fontsize=10, color="#475569")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
