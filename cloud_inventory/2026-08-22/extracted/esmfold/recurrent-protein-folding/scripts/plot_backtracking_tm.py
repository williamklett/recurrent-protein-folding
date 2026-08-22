#!/usr/bin/env python3
"""Backtracking mass vs TM-score — all proteins + failing folds subset.

Row 0: all N proteins
Row 1: only proteins with TM-score < 0.5 (fold failures)

Columns:
  0 — pLDDT baseline
  1 — raw backtracking mass
  2 — log10 backtracking mass
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "runs/general_convergence_ood500_native_cw30_stream"


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


# ── backtrack mass from layerwise CSV ─────────────────────────────────────────
def backtrack_mass(corr_p: Path) -> float:
    rows = list(csv.DictReader(corr_p.open()))
    key = "corr_vs_final_distogram" if "corr_vs_final_distogram" in rows[0] else "corr_vs_47"
    layers = np.array([int(r["layer"]) for r in rows])
    corr = np.array([float(r[key]) for r in rows])
    # drop reference row (layer 47 = 1.0 by definition)
    keep = layers < int(layers.max())
    corr = corr[keep]
    residual = np.clip(1.0 - corr, 1e-8, 1.0)
    delta = np.diff(residual)
    return float(np.maximum(delta, 0.0).sum())


# ── load data ─────────────────────────────────────────────────────────────────
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
        tm = float(s.get("tm_score", "nan"))
        plddt = float(s.get("plddt", "nan"))
        if np.isnan(tm) or np.isnan(plddt):
            continue
        bm = backtrack_mass(corr_p)
        data.append({
            "name":    name,
            "tm":      tm,
            "plddt":   plddt,
            "bt_raw":  bm,
            "bt_log":  float(np.log10(max(bm, 1e-8))),
        })
    return data


# ── scatter panel ─────────────────────────────────────────────────────────────
def tm_color(tm: float) -> str:
    if tm >= 0.7:   return "#287d52"
    if tm >= 0.5:   return "#e99a24"
    return "#c9452a"


def pearson_label(x: np.ndarray, y: np.ndarray) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return ""
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return f"r = {r:+.3f}"


def draw_panel(ax, x: np.ndarray, y: np.ndarray, names: list[str],
               xlabel: str, title: str, fail_thresh: float = 0.5) -> None:
    colors = [tm_color(tm) for tm in y]
    ax.scatter(x, y, s=95, c=colors, edgecolors="#172033", linewidths=0.6, alpha=0.92, zorder=3)
    for nm, xx, yy in zip(names, x, y):
        if np.isfinite(xx) and np.isfinite(yy):
            ax.annotate(nm, (xx, yy), fontsize=7, xytext=(4, 3),
                        textcoords="offset points", color="#1e293b")
    ax.axhline(fail_thresh, color="#0f172a", lw=0.8, ls="--", alpha=0.4)
    lbl = pearson_label(x, y)
    ax.text(0.04, 0.96, lbl, transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef3c7", ec="#a16207", alpha=0.92))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("TM-score vs PDB", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir", type=Path, default=OOD_DIR)
    ap.add_argument("--out", type=Path, default=OOD_DIR / "backtracking_vs_tm.png")
    ap.add_argument("--fail-thresh", type=float, default=0.5)
    args = ap.parse_args()

    style()
    data = load_data(args.ood_dir)
    print(f"loaded {len(data)} proteins")

    fail = [r for r in data if r["tm"] < args.fail_thresh]
    print(f"failures (TM<{args.fail_thresh}): {len(fail)}")

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.87, bottom=0.08,
                        wspace=0.28, hspace=0.40)

    fig.suptitle("Backtracking mass: raw vs log scale",
                 x=0.055, y=0.96, ha="left", fontsize=22,
                 fontweight="bold", color="#172033")
    fig.text(
        0.055, 0.915,
        f"N={len(data)} with cached coordinates.  "
        "TM is CA-coordinate TM-like score vs PDB; higher is better.  "
        "Backtracking mass = Σ max(Δresidual, 0) over all layers.",
        ha="left", fontsize=12, color="#475569",
    )

    for row_idx, (subset, row_label) in enumerate([
        (data,  f"All proteins (N={len(data)})"),
        (fail,  f"Fold failures only — TM < {args.fail_thresh} (N={len(fail)})"),
    ]):
        plddt  = np.array([r["plddt"]  for r in subset])
        bt_raw = np.array([r["bt_raw"] for r in subset])
        bt_log = np.array([r["bt_log"] for r in subset])
        tm     = np.array([r["tm"]     for r in subset])
        names  = [r["name"]            for r in subset]

        draw_panel(axes[row_idx, 0], plddt,  tm, names,
                   "ESMFold mean pLDDT",
                   f"Confidence baseline — {row_label}",
                   args.fail_thresh)
        draw_panel(axes[row_idx, 1], bt_raw, tm, names,
                   "probe backtracking mass",
                   f"Raw backtracking — {row_label}",
                   args.fail_thresh)
        draw_panel(axes[row_idx, 2], bt_log, tm, names,
                   "log10(probe backtracking mass)",
                   f"Log backtracking — {row_label}",
                   args.fail_thresh)

        # invert x on backtracking panels (higher bt = worse)
        axes[row_idx, 1].invert_xaxis()
        axes[row_idx, 2].invert_xaxis()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")

    # Print Pearson r summary
    print("\nPearson r with TM-score:")
    for subset, label in [(data, "all"), (fail, "TM<0.5")]:
        tm = np.array([r["tm"] for r in subset])
        for key in ("plddt", "bt_raw", "bt_log"):
            x = np.array([r[key] for r in subset])
            print(f"  [{label}] {key}: {pearson_label(x, tm)}")


if __name__ == "__main__":
    main()
