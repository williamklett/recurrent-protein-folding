#!/usr/bin/env python3
"""Convergence grid for the combined synthetic-design + hard-miss benchmark.
Same look as plot_benchmark_convergence_panel, but loads from two roots and
annotates each panel with the group it came from."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def load_rows(root: Path, group: str) -> list[dict]:
    rows = []
    for d in sorted(root.glob("*_analysis")):
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists():
            continue
        m = json.loads(metrics_p.read_text())
        corr_rows = list(csv.DictReader(corr_p.open()))
        layers = np.array([int(r["layer"]) for r in corr_rows])
        corr = np.array([float(r["corr_vs_47"]) for r in corr_rows])
        rows.append({
            "name": d.name.removesuffix("_analysis"),
            "group": group,
            "length": int(m["length"]),
            "plddt": float(m["mean_plddt"]),
            "mae": float(m["distance_mae_all_pairs"]),
            "layers": layers,
            "corr": corr,
            "min_last5": float(np.nanmin(corr[layers >= 43])),
            "late_amp": float(np.nanmax(corr[layers >= 36]) - np.nanmin(corr[layers >= 36])),
        })
    return rows


def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def rank_color(rank, total):
    green = np.array(mpl.colors.to_rgb("#287d52"))
    amber = np.array(mpl.colors.to_rgb("#e99a24"))
    red   = np.array(mpl.colors.to_rgb("#c9452a"))
    t = (rank - 1) / max(1, total - 1)
    if t < 0.55:
        u = t / 0.55
        return tuple(green * (1 - u) + amber * u)
    u = (t - 0.55) / 0.45
    return tuple(amber * (1 - u) + red * u)


def draw_spans(ax):
    ax.axvspan(1, 17, color="#f59e0b", alpha=0.06, lw=0)
    ax.axvspan(18, 35, color="#64748b", alpha=0.05, lw=0)
    ax.axvspan(36, 47, color="#16a34a", alpha=0.07, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct-root", type=Path, required=True)
    ap.add_argument("--miss-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Synthetic design OOD convergence benchmark")
    args = ap.parse_args()

    style()
    rows = load_rows(args.correct_root, "correct") + load_rows(args.miss_root, "miss")
    # Deduplicate by name (keep first occurrence — correct-root wins ties)
    seen = set(); deduped = []
    for r in rows:
        if r["name"] in seen: continue
        seen.add(r["name"]); deduped.append(r)
    rows = deduped
    rows.sort(key=lambda r: r["mae"])
    for i, r in enumerate(rows, 1): r["rank"] = i
    n = len(rows)
    print(f"N={n}  correct={sum(r['group']=='correct' for r in rows)}  miss={sum(r['group']=='miss' for r in rows)}")

    cols = 4
    rows_n = int(np.ceil(n / cols))
    y_min = min(0.6, min(float(np.nanmin(r["corr"])) for r in rows))
    y_max = 1.001
    fig, axes = plt.subplots(rows_n, cols, figsize=(18, 3.45 * rows_n + 2.0), sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows_n, cols)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.05, top=0.92, wspace=0.20, hspace=0.40)
    fig.suptitle(args.title, x=0.055, y=0.975, ha="left", fontsize=25, fontweight="bold", color="#172033")
    fig.text(0.055, 0.94, f"All {n} synthetic candidates. Fixed y-axis, ordered by final distance MAE.",
             ha="left", fontsize=13, color="#475569")

    for ax in axes.flat: ax.set_visible(False)
    for ax, row in zip(axes.flat, rows):
        ax.set_visible(True)
        draw_spans(ax)
        ax.plot(row["layers"], row["corr"], color="#2563eb", lw=2.2)
        ax.scatter(row["layers"], row["corr"], color="#2563eb", s=12, zorder=5)
        ax.set_xlim(1, 47); ax.set_ylim(y_min, y_max); ax.grid(True, lw=0.7)
        tag = "★ miss" if row["group"] == "miss" else ""
        ax.set_title(f"{row['rank']:02d}. {row['name']}  {tag}", loc="left", fontsize=12, pad=6)
        ax.text(0.98, 0.93,
                f"MAE {row['mae']:.2f} A\npLDDT {row['plddt']:.3f}\nlate_amp {row['late_amp']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#475569",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e2e8f0", alpha=0.9))
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color(rank_color(row["rank"], n))

    for ax in axes[-1, :]:
        if ax.get_visible(): ax.set_xlabel("ESMFold trunk block")
    for ax in axes[:, 0]:
        if ax.get_visible(): ax.set_ylabel("corr vs block 47")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
