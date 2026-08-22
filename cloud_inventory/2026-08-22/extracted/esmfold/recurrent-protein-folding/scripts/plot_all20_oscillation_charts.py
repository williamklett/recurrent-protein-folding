#!/usr/bin/env python3
"""Publication-style oscillation/convergence charts for the 20-protein panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORDER = ROOT / "runs/partial_benchmark/chimerax_cartoon_accuracy_grid_order.csv"
OUT_DIR = ROOT / "runs/partial_benchmark"


def read_order(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "name": row["name"],
                    "mae": float(row["mae"]),
                    "plddt": float(row["plddt"]),
                }
            )
    return sorted(rows, key=lambda r: r["rank"])


def read_layerwise(name: str) -> dict[str, np.ndarray]:
    path = OUT_DIR / f"{name}_analysis/layerwise_distance_correlations.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cols: dict[str, list[float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                cols.setdefault(key, []).append(float(val) if val else np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def rank_color(rank: int, n: int) -> tuple[float, float, float]:
    green = np.array(mpl.colors.to_rgb("#287d52"))
    amber = np.array(mpl.colors.to_rgb("#e99a24"))
    red = np.array(mpl.colors.to_rgb("#c9452a"))
    t = (rank - 1) / max(1, n - 1)
    if t < 0.55:
        u = t / 0.55
        return tuple(green * (1 - u) + amber * u)
    u = (t - 0.55) / 0.45
    return tuple(amber * (1 - u) + red * u)


def setup_style() -> None:
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
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )


def add_stage_spans(ax: plt.Axes) -> None:
    ax.axvspan(1, 17, color="#f59e0b", alpha=0.07, lw=0)
    ax.axvspan(18, 35, color="#64748b", alpha=0.06, lw=0)
    ax.axvspan(36, 47, color="#16a34a", alpha=0.07, lw=0)


def label_panel(ax: plt.Axes, row: dict, metric_text: str) -> None:
    ax.set_title(f"{row['rank']:02d}. {row['name']}", loc="left", pad=6)
    ax.text(
        0.98,
        0.94,
        f"MAE {row['mae']:.2f} A\npLDDT {row['plddt']:.3f}\n{metric_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e2e8f0", alpha=0.88),
    )


def grid_figure(
    rows: list[dict],
    data: dict[str, dict[str, np.ndarray]],
    y_key: str,
    out: Path,
    title: str,
    subtitle: str,
    ylabel: str,
    color: str,
    invert_y: bool = False,
    log_y: bool = False,
) -> None:
    all_y = []
    for row in rows:
        y = data[row["name"]][y_key].copy()
        if log_y:
            y = np.maximum(y, 1e-6)
        all_y.append(y[np.isfinite(y)])
    all_y_arr = np.concatenate(all_y)
    y_min = float(np.nanmin(all_y_arr))
    y_max = float(np.nanmax(all_y_arr))
    if log_y:
        y_low = 10 ** np.floor(np.log10(max(y_min, 1e-6)))
        y_high = 10 ** np.ceil(np.log10(y_max))
    else:
        pad = 0.04 * max(y_max - y_min, 1e-9)
        y_low = y_min - pad
        y_high = y_max + pad
        if y_key == "corr_vs_47":
            y_high = min(1.001, y_high)
            y_low = min(y_low, 0.6)
        if y_key == "dist_mae_vs_centroid":
            y_low = 0.0

    fig, axes = plt.subplots(5, 4, figsize=(18, 19), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.91, wspace=0.20, hspace=0.36)
    fig.suptitle(title, x=0.055, y=0.975, ha="left", fontsize=26, fontweight="bold", color="#172033")
    fig.text(0.055, 0.944, subtitle, ha="left", fontsize=13, color="#475569")

    for ax, row in zip(axes.flat, rows):
        d = data[row["name"]]
        layers = d["layer"]
        y = d[y_key].copy()
        if log_y:
            y = np.maximum(y, 1e-6)

        add_stage_spans(ax)
        ax.plot(layers, y, color=color, lw=2.1)
        ax.scatter(layers, y, color=color, s=10, zorder=5)
        ax.axvspan(43, 47, color="#22c55e", alpha=0.10, lw=0)
        ax.grid(True, lw=0.7)
        ax.set_xlim(1, 47)
        if log_y:
            ax.set_yscale("log")
        ax.set_ylim(y_low, y_high)
        if invert_y:
            ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)
            spine.set_color(rank_color(row["rank"], len(rows)))

        late = y[layers >= 36]
        amp = float(np.nanmax(late) - np.nanmin(late))
        label_panel(ax, row, f"late amp {amp:.3g}")

    for ax in axes[-1, :]:
        ax.set_xlabel("ESMFold trunk block")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


def heatmap(rows: list[dict], data: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    names = [r["name"] for r in rows]
    layers = data[names[0]]["layer"].astype(int)
    # Use -log10(1-corr) to show high-correlation oscillations that are invisible
    # on a linear correlation scale. Larger/brighter means closer to the final readout.
    mat = np.vstack([
        -np.log10(np.clip(1.0 - data[n]["corr_vs_47"], 1e-6, 1.0))
        for n in names
    ])
    mat = np.clip(mat, 0, 6)

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.subplots_adjust(left=0.13, right=0.92, bottom=0.11, top=0.80)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=6, interpolation="nearest")
    fig.text(
        0.13,
        0.935,
        "Layerwise convergence/oscillation map",
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.13,
        0.895,
        "Rows are ordered by final structural error, accurate at top and inaccurate at bottom.",
        ha="left",
        va="top",
        fontsize=13,
        color="#475569",
    )
    fig.text(
        0.13,
        0.865,
        "Color is -log10(1 - corr to block 47): brighter means the readout is closer to the final distogram. Rows are accurate -> inaccurate.",
        ha="left",
        va="top",
        fontsize=11,
        color="#475569",
    )
    ax.set_xticks(np.arange(0, len(layers), 4), labels=[str(x) for x in layers[::4]])
    ax.set_yticks(np.arange(len(names)), labels=[f"{r['rank']:02d}. {r['name']}  MAE {r['mae']:.2f}" for r in rows])
    ax.set_xlabel("ESMFold trunk block")
    ax.set_ylabel("Protein, sorted by final distance MAE")
    for boundary in [16.5, 34.5]:
        ax.axvline(boundary, color="white", lw=1.2, alpha=0.65)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.set_label("-log10(1 - corr vs final readout)")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


def summary_csv(rows: list[dict], data: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    with out.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "name",
            "distance_mae",
            "plddt",
            "min_corr_last5",
            "late_corr_amp_36_47",
            "late_centroid_amp_36_47",
            "early_corr_amp_1_17",
            "mid_corr_amp_18_35",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = data[row["name"]]
            layers = d["layer"]
            corr = d["corr_vs_47"]
            centroid = d["dist_mae_vs_centroid"]
            def amp(mask: np.ndarray, arr: np.ndarray) -> float:
                v = arr[mask]
                return float(np.nanmax(v) - np.nanmin(v))
            writer.writerow(
                {
                    "rank": row["rank"],
                    "name": row["name"],
                    "distance_mae": f"{row['mae']:.6g}",
                    "plddt": f"{row['plddt']:.6g}",
                    "min_corr_last5": f"{np.nanmin(corr[layers >= 43]):.6g}",
                    "late_corr_amp_36_47": f"{amp(layers >= 36, corr):.6g}",
                    "late_centroid_amp_36_47": f"{amp(layers >= 36, centroid):.6g}",
                    "early_corr_amp_1_17": f"{amp(layers <= 17, corr):.6g}",
                    "mid_corr_amp_18_35": f"{amp((layers >= 18) & (layers <= 35), corr):.6g}",
                }
            )
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-csv", type=Path, default=DEFAULT_ORDER)
    args = ap.parse_args()

    setup_style()
    rows = read_order(args.order_csv)
    data = {row["name"]: read_layerwise(row["name"]) for row in rows}

    grid_figure(
        rows,
        data,
        y_key="corr_vs_47",
        out=OUT_DIR / "all20_oscillation_corr_vs_final_grid.png",
        title="Layerwise distogram convergence, ordered by structural error",
        subtitle="Accurate proteins show high-correlation refinement and late small-amplitude motion; failures remain lower, driftier, or collapse late.",
        ylabel="corr vs block 47",
        color="#2563eb",
    )

    grid_figure(
        rows,
        data,
        y_key="dist_mae_vs_centroid",
        out=OUT_DIR / "all20_oscillation_centroid_distance_grid.png",
        title="Attractor/orbit diagnostic across all 20 proteins",
        subtitle="Distance to each protein's layerwise centroid: bounded late motion reads like orbiting an attractor; failure cases show larger excursions or weak settling.",
        ylabel="distance to trajectory centroid (A)",
        color="#7c3aed",
    )

    # The error plot makes tiny late oscillations visible without pretending that
    # absolute correlations differ on the same visual scale.
    for row in rows:
        data[row["name"]]["one_minus_corr"] = np.clip(1.0 - data[row["name"]]["corr_vs_47"], 1e-6, 1.0)
    grid_figure(
        rows,
        data,
        y_key="one_minus_corr",
        out=OUT_DIR / "all20_oscillation_error_to_final_grid.png",
        title="Residual error to final readout exposes the oscillation scale",
        subtitle="Same signal as correlation-to-final, plotted as 1 - corr on a log scale. Down is closer to the final readout.",
        ylabel="1 - corr vs block 47",
        color="#dc2626",
        invert_y=True,
        log_y=True,
    )

    heatmap(rows, data, OUT_DIR / "all20_oscillation_convergence_heatmap.png")
    summary_csv(rows, data, OUT_DIR / "all20_oscillation_summary.csv")


if __name__ == "__main__":
    main()
