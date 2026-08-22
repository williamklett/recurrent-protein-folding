#!/usr/bin/env python3
"""Plot convergence/performance charts for a benchmark directory.

Expected layout:
  ROOT/<name>_analysis/
    layerwise_distance_correlations.csv
    esmfold_direct_eval/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def metrics_for_name(name: str, root: Path) -> dict | None:
    candidates = [
        root / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / f"benchmark_{name}" / "analysis/esmfold_direct_eval/metrics.json",
        Path("runs") / f"hard_synthetic_miss_{name}" / "analysis/esmfold_direct_eval/metrics.json",
        Path("runs") / "general_convergence_n40_native_cw30_partial" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "general_convergence_n40_native" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "general_convergence_hard_synthetic_native" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "partial_benchmark" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "hard_synthetic_miss_partial_sync" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "synthetic_design_partial_sync" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
    ]
    p = next((p for p in candidates if p.exists()), None)
    return json.loads(p.read_text()) if p else None


def row_from_csv(name: str, corr_p: Path, metrics: dict, ref_label: str) -> dict:
    corr_rows = list(csv.DictReader(corr_p.open()))
    layers = np.array([int(r["layer"]) for r in corr_rows])
    corr_key = "corr_vs_final_distogram" if "corr_vs_final_distogram" in corr_rows[0] else "corr_vs_47"
    corr = np.array([float(r[corr_key]) for r in corr_rows])
    centroid = np.array([float(r["dist_mae_vs_centroid"]) for r in corr_rows])
    probe_layers = layers[layers < int(np.nanmax(layers))]
    late5_start = int(np.nanmax(probe_layers)) - 4 if len(probe_layers) else int(np.nanmax(layers)) - 4
    late_start = max(1, int(np.nanmax(probe_layers)) - 11) if len(probe_layers) else max(1, int(np.nanmax(layers)) - 11)
    return {
        "name": name,
        "length": int(metrics["length"]),
        "plddt": float(metrics["mean_plddt"]),
        "rmsd": float(metrics["ca_rmsd_after_kabsch"]),
        "mae": float(metrics["distance_mae_all_pairs"]),
        "dist_corr": float(metrics["distance_corr_all_pairs"]),
        "layers": layers,
        "corr": corr,
        "centroid": centroid,
        "ref_label": ref_label,
        "min_last5": float(np.nanmin(corr[(layers >= late5_start) & (layers < np.nanmax(layers))])),
        "late_amp": float(np.nanmax(corr[(layers >= late_start) & (layers < np.nanmax(layers))]) - np.nanmin(corr[(layers >= late_start) & (layers < np.nanmax(layers))])),
        "centroid_late_amp": float(np.nanmax(centroid[(layers >= late_start) & (layers < np.nanmax(layers))]) - np.nanmin(centroid[(layers >= late_start) & (layers < np.nanmax(layers))])),
    }


def load_rows(root: Path) -> list[dict]:
    rows = []
    flat_csvs = sorted(root.glob("*_layerwise.csv"))
    if flat_csvs:
        for corr_p in flat_csvs:
            name = corr_p.name.removesuffix("_layerwise.csv")
            m = metrics_for_name(name, root)
            if m is None:
                continue
            rows.append(row_from_csv(name, corr_p, m, "native ESMFold output"))
        rows.sort(key=lambda r: r["mae"])
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        return rows

    for d in sorted(root.glob("*_analysis")):
        name = d.name.removesuffix("_analysis")
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists():
            continue
        m = json.loads(metrics_p.read_text())
        corr_rows = list(csv.DictReader(corr_p.open()))
        ref_label = "trained final distogram" if "corr_vs_final_distogram" in corr_rows[0] else "native ESMFold output"
        rows.append(row_from_csv(name, corr_p, m, ref_label))
    rows.sort(key=lambda r: r["mae"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#e2e8f0",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def rank_color(rank: int, total: int) -> tuple[float, float, float]:
    green = np.array(mpl.colors.to_rgb("#287d52"))
    amber = np.array(mpl.colors.to_rgb("#e99a24"))
    red = np.array(mpl.colors.to_rgb("#c9452a"))
    t = (rank - 1) / max(1, total - 1)
    if t < 0.55:
        u = t / 0.55
        return tuple(green * (1 - u) + amber * u)
    u = (t - 0.55) / 0.45
    return tuple(amber * (1 - u) + red * u)


def draw_spans(ax: plt.Axes) -> None:
    ax.axvspan(1, 17, color="#f59e0b", alpha=0.06, lw=0)
    ax.axvspan(18, 35, color="#64748b", alpha=0.05, lw=0)
    ax.axvspan(35, 46, color="#16a34a", alpha=0.07, lw=0)


def convergence_grid(rows: list[dict], out: Path, title: str) -> None:
    n = len(rows)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    max_layer = max(int(np.nanmax(r["layers"])) for r in rows)
    y_min = min(0.6, min(float(np.nanmin(r["corr"])) for r in rows))
    y_max = 1.001
    fig, axes = plt.subplots(rows_n, cols, figsize=(18, 3.45 * rows_n + 2.0), sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows_n, cols)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.06, top=0.88, wspace=0.20, hspace=0.40)
    fig.suptitle(title, x=0.055, y=0.975, ha="left", fontsize=25, fontweight="bold", color="#172033")
    fig.text(
        0.055,
        0.935,
        "Fixed y-axis across proteins. Ordered by final distance MAE, most accurate to least accurate.",
        ha="left",
        fontsize=13,
        color="#475569",
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, row in zip(axes.flat, rows):
        ax.set_visible(True)
        draw_spans(ax)
        ax.plot(row["layers"], row["corr"], color="#2563eb", lw=2.2)
        ax.scatter(row["layers"], row["corr"], color="#2563eb", s=12, zorder=5)
        ax.set_xlim(1, max_layer)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, lw=0.7)
        ax.set_title(f"{row['rank']:02d}. {row['name']}", loc="left", fontsize=12, pad=6)
        ax.text(
            0.98,
            0.93,
            f"MAE {row['mae']:.2f} A\npLDDT {row['plddt']:.3f}\nmin last5 {row['min_last5']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#475569",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e2e8f0", alpha=0.9),
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color(rank_color(row["rank"], len(rows)))
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("ESMFold trunk block")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(f"corr vs {rows[0].get('ref_label', 'reference')}")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def scatter_summary(rows: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.14, top=0.80, wspace=0.28)
    fig.suptitle(title, x=0.055, y=0.96, ha="left", fontsize=24, fontweight="bold", color="#172033")
    fig.text(0.055, 0.89, f"N={len(rows)} cached proteins. Correlations are vs {rows[0].get('ref_label', 'reference')}.", ha="left", fontsize=13, color="#475569")
    names = [r["name"] for r in rows]
    mae = np.array([r["mae"] for r in rows])
    mae_log = np.log10(np.clip(mae, 1e-6, None))
    plddt = np.array([r["plddt"] for r in rows])
    conv = np.log10(np.clip(1.0 - np.array([r["min_last5"] for r in rows]), 1e-6, 1.0))
    late_amp = np.log10(np.clip(np.array([r["late_amp"] for r in rows]), 1e-6, None))
    colors = [rank_color(r["rank"], len(rows)) for r in rows]

    panels = [
        (plddt, mae_log, "ESMFold mean pLDDT", "log10(distance MAE vs PDB)", "pLDDT calibration"),
        (conv, mae_log, "log10(1 - min corr), final 5 probe readouts", "log10(distance MAE vs PDB)", "convergence classifier"),
        (late_amp, mae_log, "log10(late corr amplitude), final 12 probe readouts", "log10(distance MAE vs PDB)", "instability classifier"),
    ]
    for ax, (x, y, xlabel, ylabel, subtitle) in zip(axes, panels):
        ax.scatter(x, y, s=115, c=colors, edgecolors="#172033", linewidths=0.7)
        for name, xx, yy in zip(names, x, y):
            ax.annotate(name, (xx, yy), fontsize=8, xytext=(4, 4), textcoords="offset points")
        if len(rows) > 2 and np.nanstd(x) > 1e-9 and np.nanstd(y) > 1e-9:
            rr = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.04, 0.94, f"r = {rr:+.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.7)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(rows: list[dict], out: Path) -> None:
    fields = ["rank", "name", "length", "mae", "rmsd", "dist_corr", "plddt", "min_last5", "late_amp", "centroid_late_amp"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--title", default="Synthetic design OOD convergence benchmark")
    args = ap.parse_args()
    style()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"No completed runs in {args.root}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    convergence_grid(rows, args.out_prefix.with_name(args.out_prefix.name + "_convergence_grid.png"), args.title)
    scatter_summary(rows, args.out_prefix.with_name(args.out_prefix.name + "_performance_scatter.png"), args.title)
    write_summary(rows, args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"))
    print(f"N={len(rows)}")
    for r in rows:
        print(f"{r['rank']:02d} {r['name']:>6} MAE={r['mae']:.2f} pLDDT={r['plddt']:.3f} min_last5={r['min_last5']:.3f} late_amp={r['late_amp']:.4f}")


if __name__ == "__main__":
    main()
