#!/usr/bin/env python3
"""PCE metrics vs TM-score — all proteins + failing folds subset.

Row 0: all N proteins
Row 1: only proteins with TM-score < 0.5 (fold failures)

Columns:
  0 — pLDDT baseline
  1 — pce_raw  (Kadane max positive curvature on raw d²)
  2 — pce_smooth σ=1.5 (same on Gaussian-smoothed curve)
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


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def kadane_max(arr: np.ndarray) -> float:
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def pce_raw(corr: np.ndarray) -> float:
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))


def pce_smooth(corr: np.ndarray, sigma: float = 1.5) -> float:
    c_sm = gaussian_filter1d(corr.astype(float), sigma=sigma)
    d2 = np.diff(c_sm, n=2)
    return kadane_max(np.maximum(d2, 0.0))


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
        tm       = float(s.get("tm_score",  "nan"))
        plddt    = float(s.get("plddt",     "nan"))
        min_last5 = float(s.get("min_last5", "nan"))
        if np.isnan(tm) or np.isnan(plddt):
            continue

        rows   = list(csv.DictReader(corr_p.open()))
        key    = "corr_vs_final_distogram" if "corr_vs_final_distogram" in rows[0] else "corr_vs_47"
        layers = np.array([int(r["layer"]) for r in rows])
        corr   = np.array([float(r[key])   for r in rows])
        corr46 = corr[layers <= 46]          # exclude reference row

        data.append({
            "name":       name,
            "tm":         tm,
            "plddt":      plddt,
            "pce_raw":    pce_raw(corr46),
            "pce_smooth": pce_smooth(corr46, sigma=1.5),
            "min_last5":  min_last5,
        })
    return data


def tm_color(tm: float) -> str:
    if tm >= 0.7: return "#287d52"
    if tm >= 0.5: return "#e99a24"
    return "#c9452a"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3: return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def draw_panel(ax, x, y, names, xlabel, title, fail_thresh=0.5):
    colors = [tm_color(tm) for tm in y]
    ax.scatter(x, y, s=95, c=colors, edgecolors="#172033",
               linewidths=0.6, alpha=0.92, zorder=3)
    for nm, xx, yy in zip(names, x, y):
        if np.isfinite(xx) and np.isfinite(yy):
            ax.annotate(nm, (xx, yy), fontsize=7, xytext=(4, 3),
                        textcoords="offset points", color="#1e293b")
    ax.axhline(fail_thresh, color="#0f172a", lw=0.8, ls="--", alpha=0.4)
    r = pearson_r(np.asarray(x, float), np.asarray(y, float))
    ax.text(0.04, 0.96, f"r = {r:+.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef3c7", ec="#a16207", alpha=0.92))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("TM-score vs PDB", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir",     type=Path, default=OOD_DIR)
    ap.add_argument("--out",         type=Path, default=OOD_DIR / "pce_vs_tm.png")
    ap.add_argument("--fail-thresh", type=float, default=0.5)
    args = ap.parse_args()

    style()
    data = load_data(args.ood_dir)
    fail = [r for r in data if r["tm"] < args.fail_thresh]
    print(f"N={len(data)}  failures(TM<{args.fail_thresh})={len(fail)}")

    fig, axes = plt.subplots(2, 4, figsize=(25, 12))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.87, bottom=0.08,
                        wspace=0.28, hspace=0.40)
    fig.suptitle("PCE + min_last5 metrics vs TM-score",
                 x=0.045, y=0.96, ha="left", fontsize=22,
                 fontweight="bold", color="#172033")
    fig.text(
        0.045, 0.915,
        f"N={len(data)} OOD-500 proteins (cw30 probes).  "
        "pce = Kadane max integral of positive second-derivative of corr_vs_47.  "
        "min_last5 = min corr_vs_47 over layers 42–46.",
        ha="left", fontsize=12, color="#475569",
    )

    panels = [
        ("plddt",      "ESMFold mean pLDDT",                        "Confidence baseline"),
        ("pce_raw",    "pce_raw (Kadane on raw d²)",                 "PCE raw"),
        ("pce_smooth", "pce_smooth σ=1.5 (Kadane on smoothed d²)",  "PCE smooth"),
        ("min_last5",  "min corr_vs_47 (layers 42–46)",             "min_last5"),
    ]

    for row_idx, (subset, row_label) in enumerate([
        (data, f"All proteins  (N={len(data)})"),
        (fail, f"Fold failures only — TM < {args.fail_thresh}  (N={len(fail)})"),
    ]):
        tm    = np.array([r["tm"]    for r in subset])
        names = [r["name"]           for r in subset]
        for col, (key, xlabel, col_title) in enumerate(panels):
            x = np.array([r[key] for r in subset])
            draw_panel(axes[row_idx, col], x, tm, names,
                       xlabel, f"{col_title} — {row_label}",
                       args.fail_thresh)
            # pce panels: invert x so rightward = less curvature = better
            # min_last5 is higher=better like pLDDT so no inversion
            if key in ("pce_raw", "pce_smooth"):
                axes[row_idx, col].invert_xaxis()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")

    print("\nPearson r with TM-score:")
    for subset, label in [(data, "all"), (fail, "TM<0.5")]:
        tm = np.array([r["tm"] for r in subset])
        for key in ("plddt", "pce_raw", "pce_smooth", "min_last5"):
            x  = np.array([r[key] for r in subset])
            r  = pearson_r(x, tm)
            print(f"  [{label:7s}] {key:12s}: r = {r:+.3f}")


if __name__ == "__main__":
    main()
