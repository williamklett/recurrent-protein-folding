#!/usr/bin/env python3
"""Render a layerwise MDS PDB stack as a fixed-camera folding trajectory."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Line3DCollection


SEC_COLORS = {
    "H": "#d95f02",
    "E": "#1b77b4",
    "C": "#8c939c",
}

INK = "#17202a"
MUTED = "#596575"
GRID = "#d9dee7"
ACCENT = "#c84a31"
SUCCESS = "#2f7d55"


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = min(len(mobile), len(target))
    m = mobile[:n]
    t = target[:n]
    mc = m - m.mean(axis=0, keepdims=True)
    tc = t - t.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(mc.T @ tc)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = (mobile - m.mean(axis=0, keepdims=True)) @ rot + t.mean(axis=0, keepdims=True)
    return aligned


def parse_models(path: Path) -> dict[int, np.ndarray]:
    models: dict[int, np.ndarray] = {}
    layer = None
    coords: list[tuple[float, float, float]] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("MODEL"):
                parts = line.split()
                layer = int(parts[-1])
                coords = []
            elif line.startswith("ATOM") and layer is not None:
                coords.append(
                    (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
            elif line.startswith("ENDMDL") and layer is not None:
                models[layer] = np.asarray(coords, dtype=np.float32)
                layer = None
    if not models:
        raise ValueError(f"no MODEL blocks found in {path}")
    return models


def parse_ca_and_secondary(path: Path, chain: str = "A") -> tuple[np.ndarray, list[str]]:
    helix_ranges: list[tuple[int, int]] = []
    sheet_ranges: list[tuple[int, int]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("HELIX") and line[19].strip() == chain:
            helix_ranges.append((int(line[21:25]), int(line[33:37])))
        elif line.startswith("SHEET") and line[21].strip() == chain:
            sheet_ranges.append((int(line[22:26]), int(line[33:37])))

    coords: list[tuple[float, float, float]] = []
    labels: list[str] = []
    seen: set[tuple[int, str]] = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain):
                continue
            resnum = int(line[22:26])
            ins = line[26]
            key = (resnum, ins)
            if key in seen:
                continue
            seen.add(key)
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            sec = "C"
            if any(a <= resnum <= b for a, b in helix_ranges):
                sec = "H"
            elif any(a <= resnum <= b for a, b in sheet_ranges):
                sec = "E"
            labels.append(sec)
    if not coords:
        raise ValueError(f"no CA atoms found for chain {chain!r} in {path}")
    return np.asarray(coords, dtype=np.float32), labels


def interpolate_models(models: dict[int, np.ndarray], frames_per_step: int) -> list[tuple[float, np.ndarray]]:
    layers = sorted(models)
    frames: list[tuple[float, np.ndarray]] = []
    for a, b in zip(layers[:-1], layers[1:]):
        ca, cb = models[a], models[b]
        for k in range(frames_per_step):
            t = k / frames_per_step
            frames.append((a + t, ca * (1.0 - t) + cb * t))
    frames.append((float(layers[-1]), models[layers[-1]]))
    return frames


def equal_axes(ax, coords: np.ndarray, margin: float = 0.08) -> None:
    lo = coords.min(axis=(0, 1))
    hi = coords.max(axis=(0, 1))
    center = (lo + hi) / 2.0
    span = float((hi - lo).max()) * (1.0 + margin)
    if span <= 0:
        span = 1.0
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)


def load_summary(run_dir: Path) -> tuple[str, str]:
    metrics_path = run_dir / "esmfold_direct_eval" / "metrics.json"
    corr_path = run_dir / "layerwise_distance_correlations.csv"
    if not metrics_path.exists() and (run_dir / "analysis" / "esmfold_direct_eval" / "metrics.json").exists():
        metrics_path = run_dir / "analysis" / "esmfold_direct_eval" / "metrics.json"
    if not corr_path.exists() and (run_dir / "analysis" / "layerwise_distance_correlations.csv").exists():
        corr_path = run_dir / "analysis" / "layerwise_distance_correlations.csv"
    left = ""
    right = ""
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        left = (
            f"MAE {metrics['distance_mae_all_pairs']:.2f} A   "
            f"pLDDT {metrics['mean_plddt']:.3f}"
        )
    if corr_path.exists():
        rows = {int(r["layer"]): r for r in csv.DictReader(corr_path.open())}
        late = [
            float(rows[layer]["corr_vs_47"])
            for layer in range(36, 48)
            if layer in rows
        ]
        if late:
            right = f"min late convergence {min(late):.4f}"
    return left, right


def load_layer_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    corr_path = run_dir / "layerwise_distance_correlations.csv"
    if not corr_path.exists() and (run_dir / "analysis" / "layerwise_distance_correlations.csv").exists():
        corr_path = run_dir / "analysis" / "layerwise_distance_correlations.csv"
    if not corr_path.exists():
        return {}
    rows: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(corr_path.open()):
        layer = int(row["layer"])
        rows[layer] = {}
        for key, value in row.items():
            if key == "layer" or value == "":
                continue
            try:
                rows[layer][key] = float(value)
            except ValueError:
                continue
    return rows


def segment_collection(
    coords: np.ndarray,
    sec_labels: list[str],
    linewidth: float,
    alpha: float,
) -> Line3DCollection:
    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    colors = [SEC_COLORS.get(sec_labels[min(i, len(sec_labels) - 1)], SEC_COLORS["C"]) for i in range(len(segments))]
    collection = Line3DCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
    return collection


def draw_frame(
    coords: np.ndarray,
    all_coords: np.ndarray,
    layer: float,
    title: str,
    summary_left: str,
    summary_right: str,
    sec_labels: list[str],
    final_coords: np.ndarray,
    truth_coords: np.ndarray | None,
    show_truth_ghost: bool,
    layer_metrics: dict[int, dict[str, float]],
    width: int,
    height: int,
    elev: float,
    azim: float,
    publication: bool,
) -> np.ndarray:
    if publication:
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        gs = fig.add_gridspec(
            2,
            5,
            width_ratios=[1.18, 1.18, 1.18, 0.95, 0.95],
            height_ratios=[1.02, 0.98],
            left=0.045,
            right=0.965,
            top=0.89,
            bottom=0.075,
            wspace=0.22,
            hspace=0.22,
        )
        ax = fig.add_subplot(gs[:, :3], projection="3d")
        ax_chart = fig.add_subplot(gs[0, 3:])
        ax_text = fig.add_subplot(gs[1, 3:])
    else:
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax_chart = None
        ax_text = None
    n = len(coords)
    display_coords = [all_coords, final_coords[None, :, :]]
    if truth_coords is not None and show_truth_ghost:
        display_coords.append(truth_coords[None, :, :])
    ax.add_collection3d(segment_collection(final_coords, sec_labels, linewidth=2.6, alpha=0.16))
    if truth_coords is not None and show_truth_ghost:
        ax.plot(
            truth_coords[:, 0],
            truth_coords[:, 1],
            truth_coords[:, 2],
            color="#111827",
            lw=1.0,
            alpha=0.18,
        )
    ax.add_collection3d(segment_collection(coords, sec_labels, linewidth=4.8 if publication else 3.0, alpha=0.98))
    ax.scatter(coords[[0, -1], 0], coords[[0, -1], 1], coords[[0, -1], 2], s=52, c=[INK, "#f2c94c"], depthshade=True)
    ax.text(coords[0, 0], coords[0, 1], coords[0, 2], "N", fontsize=10, color=INK, weight="bold")
    ax.text(coords[-1, 0], coords[-1, 1], coords[-1, 2], "C", fontsize=10, color=INK, weight="bold")
    equal_axes(ax, np.concatenate(display_coords, axis=0), margin=0.18 if publication else 0.08)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("#fafbfc")
    fig.patch.set_facecolor("#fafbfc")
    if publication:
        fig.suptitle(title, x=0.045, y=0.965, ha="left", fontsize=21, fontweight="bold", color=INK)
        fig.text(
            0.045,
            0.925,
            "Decoded distogram trajectory as a C-alpha tube; helix/sheet colors are sequence annotations",
            fontsize=12,
            color=MUTED,
        )
    else:
        fig.suptitle(title, x=0.05, y=0.965, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.05, 0.92, f"Decoded MDS structure, block {layer:05.2f}", fontsize=11)

    if publication and ax_chart is not None and layer_metrics:
        layers = sorted(layer_metrics)
        y = [layer_metrics[l].get("corr_vs_47", np.nan) for l in layers]
        y_min = min(y)
        y_floor = max(0.0, y_min - 0.018)
        ax_chart.axvspan(36, 47, color="#eaf4ef", alpha=0.9, lw=0)
        ax_chart.fill_between(layers, y, y_floor, color="#dce9f7", alpha=0.55)
        ax_chart.plot(layers, y, color=INK, lw=2.4, solid_capstyle="round")
        ax_chart.scatter(layers, y, s=20, color=INK, alpha=0.62, zorder=3)
        current_layer = int(round(layer))
        if current_layer in layer_metrics:
            current_y = layer_metrics[current_layer].get("corr_vs_47", np.nan)
            ax_chart.scatter([current_layer], [current_y], s=112, color=ACCENT, zorder=5, edgecolor="white", linewidth=1.2)
            ax_chart.axvline(current_layer, color=ACCENT, lw=1.3, alpha=0.45)
        ax_chart.set_xlim(min(layers), max(layers))
        ax_chart.set_ylim(y_floor, 1.002)
        ax_chart.set_title("Convergence trajectory", fontsize=13, loc="left", color=INK, pad=8, weight="bold")
        ax_chart.text(
            46.7,
            y_floor + (1.002 - y_floor) * 0.07,
            "late window",
            ha="right",
            va="bottom",
            color=SUCCESS,
            fontsize=9,
            alpha=0.9,
        )
        ax_chart.set_xlabel("ESMFold block", color=MUTED)
        ax_chart.set_ylabel("distance-readout corr. vs final", color=MUTED)
        ax_chart.grid(color=GRID, linewidth=0.8, alpha=0.72)
        ax_chart.tick_params(colors=MUTED, labelsize=9)
        ax_chart.spines[["top", "right"]].set_visible(False)
        ax_chart.spines[["left", "bottom"]].set_color("#b7bfcc")

    if publication and ax_text is not None:
        ax_text.axis("off")
        current_layer = int(round(layer))
        metric = layer_metrics.get(current_layer, {})
        label_color = SUCCESS if "MAE" in summary_left and "pLDDT" in summary_left else INK
        metric_lines = [
            f"Block {layer:05.2f}",
            summary_left,
            summary_right,
        ]
        if metric:
            metric_lines.extend(
                [
                    f"corr vs final {metric.get('corr_vs_47', float('nan')):.4f}",
                    f"RMSD vs final {metric.get('coord_rmsd_vs_47', 0.0):.2f} A",
                ]
            )
        ax_text.text(
            0.0,
            0.96,
            "\n".join([line for line in metric_lines if line]),
            va="top",
            fontsize=13,
            color=INK,
            linespacing=1.52,
        )
        ax_text.text(0.0, 0.24, "secondary-structure annotation", fontsize=10.5, color=MUTED, weight="bold")
        ax_text.text(0.0, 0.16, "orange  helix\nblue    beta strand\ngray    loop / other", va="top", fontsize=11, color=MUTED, linespacing=1.55)
    else:
        if summary_left:
            fig.text(0.05, 0.055, summary_left, fontsize=11)
        if summary_right:
            fig.text(0.95, 0.055, summary_right, fontsize=11, ha="right")
    if not publication:
        fig.tight_layout(pad=0.0)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pdb", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--ground-truth-pdb", type=Path)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--show-truth-ghost", action="store_true")
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--interp-frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1008)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    args = parser.parse_args()

    pdb = args.pdb or args.run_dir / "mds_structure_video_fixed" / "distogram_mds_models.pdb"
    out_dir = args.out_dir or args.run_dir / "mds_structure_video_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_models(pdb)
    final_layer = max(models)
    final_coords = models[final_layer]
    truth_coords = None
    sec_labels = ["C"] * len(final_coords)
    if args.ground_truth_pdb:
        truth_raw, sec_raw = parse_ca_and_secondary(args.ground_truth_pdb, args.chain)
        n = min(len(final_coords), len(truth_raw), len(sec_raw))
        truth_coords = kabsch_align(truth_raw[:n], final_coords[:n])
        sec_labels = sec_raw[:n]
        models = {layer: coords[:n] for layer, coords in models.items()}
        final_coords = models[final_layer][:n]
    models = {layer: kabsch_align(coords, final_coords) for layer, coords in models.items()}
    final_coords = models[final_layer]
    frames = interpolate_models(models, args.interp_frames)
    all_coords = np.stack([coords for _, coords in frames], axis=0)
    title = args.title or args.run_dir.name.replace("_analysis", "")
    summary_left, summary_right = load_summary(args.run_dir)
    layer_metrics = load_layer_metrics(args.run_dir)

    rendered = [
        draw_frame(
            coords,
            all_coords,
            layer,
            title,
            summary_left,
            summary_right,
            sec_labels,
            final_coords,
            truth_coords,
            args.show_truth_ghost,
            layer_metrics,
            args.width,
            args.height,
            args.elev,
            args.azim,
            args.publication,
        )
        for layer, coords in frames
    ]

    stem = "distogram_mds_structure_publication" if args.publication else "distogram_mds_structure"
    mp4 = out_dir / f"{stem}.mp4"
    gif = out_dir / f"{stem}.gif"
    png = out_dir / f"{stem}_final.png"
    iio.imwrite(mp4, np.stack(rendered), fps=args.fps, codec="libx264", quality=8)
    iio.imwrite(gif, np.stack(rendered), fps=args.fps)
    Image.fromarray(rendered[-1]).save(png)
    print(mp4)
    print(gif)
    print(png)


if __name__ == "__main__":
    main()
