#!/usr/bin/env python3
"""Inspect reconstruction/consistency tradeoffs in general probe training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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


def read_layer(root: Path, layer: int) -> list[dict]:
    p = root / f"layer_{layer:02d}/metrics.csv"
    out = []
    for row in csv.DictReader(p.open()):
        out.append(
            {
                "step": float(row["step"]),
                "kl": float(row["kl_to_target"]),
                "recon_mse": float(row["recon_mse"]),
                "recon_r2": float(row["recon_r2"]),
                "ed_diff": float(row["ed_diff"]),
                "top1": float(row["match_top1"]),
            }
        )
    return out


def final_summary(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.glob("layer_*/metrics.csv")):
        layer = int(p.parent.name.split("_")[-1])
        data = list(csv.DictReader(p.open()))
        if not data:
            continue
        r = data[-1]
        rows.append(
            {
                "layer": layer,
                "kl": float(r["kl_to_target"]),
                "recon_mse": float(r["recon_mse"]),
                "recon_r2": float(r["recon_r2"]),
                "ed_diff": float(r["ed_diff"]),
                "top1": float(r["match_top1"]),
            }
        )
    return sorted(rows, key=lambda r: r["layer"])


def plot(root: Path, out: Path) -> None:
    style()
    layers_to_show = [44, 45, 46]
    histories = {layer: read_layer(root, layer) for layer in layers_to_show}
    final = final_summary(root)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.08, top=0.86, wspace=0.24, hspace=0.32)
    fig.suptitle("General Probe Reconstruction vs Consistency Tradeoff", x=0.065, y=0.965, ha="left", fontsize=24, fontweight="bold", color="#172033")
    fig.text(0.065, 0.915, "Layer 46 is anchored to the native ESMFold distogram head; layer 45 is anchored to the trained layer-46 probe.", ha="left", fontsize=13, color="#475569")

    colors = {44: "#64748b", 45: "#2563eb", 46: "#c9452a"}
    for layer, hist in histories.items():
        x = np.array([r["step"] for r in hist])
        axes[0, 0].plot(x, [r["recon_mse"] for r in hist], lw=2.2, label=f"L{layer}", color=colors[layer])
        axes[0, 1].plot(x, [r["kl"] for r in hist], lw=2.2, label=f"L{layer}", color=colors[layer])
        axes[1, 0].plot(x, [r["ed_diff"] for r in hist], lw=2.2, label=f"L{layer}", color=colors[layer])
    for ax, title, ylabel in [
        (axes[0, 0], "Reconstruction MSE during training", "MSE"),
        (axes[0, 1], "Consistency KL during training", "KL"),
        (axes[1, 0], "Expected-distance mismatch during training", "A"),
    ]:
        ax.set_title(title, loc="left")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.8)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    x = np.array([r["recon_r2"] for r in final])
    y = np.array([r["ed_diff"] for r in final])
    layer_nums = [r["layer"] for r in final]
    ax.scatter(x, y, s=55, color="#334155", alpha=0.82, edgecolors="white", linewidth=0.7)
    for layer, xx, yy in zip(layer_nums, x, y):
        if layer in [44, 45, 46] or yy > 1.0:
            ax.annotate(str(layer), (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.scatter(
        [final[layer_nums.index(46)]["recon_r2"]],
        [final[layer_nums.index(46)]["ed_diff"]],
        s=130,
        color="#c9452a",
        edgecolors="white",
        linewidth=1.2,
        zorder=5,
        label="L46 native-head anchor",
    )
    ax.set_title("Final per-block tradeoff", loc="left")
    ax.set_xlabel("reconstruction R²")
    ax.set_ylabel("expected-distance mismatch (A)")
    ax.grid(True, lw=0.8)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/general_belief_replay_hard20_metrics"))
    parser.add_argument("--out", type=Path, default=Path("runs/general_belief_replay_hard20_metrics/general_probe_tradeoff_inspection.png"))
    args = parser.parse_args()
    plot(args.root, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
