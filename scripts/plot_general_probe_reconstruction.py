#!/usr/bin/env python3
"""Plot reconstruction metrics for the shared general distogram probes."""

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


def load_rows(root: Path) -> list[dict]:
    rows = []
    for metrics in sorted(root.glob("layer_*/metrics.csv")):
        layer = int(metrics.parent.name.split("_")[-1])
        data = list(csv.DictReader(metrics.open()))
        if not data:
            continue
        final = data[-1]
        values = {
            "layer": layer,
            "step": int(float(final["step"])),
            "loss": float(final["loss"]),
            "kl_to_target": float(final["kl_to_target"]),
            "recon_mse": float(final["recon_mse"]),
            "recon_r2": float(final["recon_r2"]),
            "ed_diff": float(final["ed_diff"]),
            "match_top1": float(final["match_top1"]),
        }
        rows.append(values)
    return sorted(rows, key=lambda r: r["layer"])


def write_summary(rows: list[dict], out: Path) -> None:
    with out.open("w", newline="") as f:
        fields = ["layer", "step", "loss", "kl_to_target", "recon_mse", "recon_r2", "ed_diff", "match_top1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], out: Path) -> None:
    layers = np.array([r["layer"] for r in rows])
    recon_mse = np.array([r["recon_mse"] for r in rows])
    recon_r2 = np.array([r["recon_r2"] for r in rows])
    kl = np.array([r["kl_to_target"] for r in rows])
    ed = np.array([r["ed_diff"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.08, top=0.86, wspace=0.22, hspace=0.30)
    fig.suptitle("General Probe Training Metrics by ESMFold Block", x=0.065, y=0.965, ha="left", fontsize=24, fontweight="bold", color="#172033")
    fig.text(0.065, 0.915, "Final recorded training row per block from runs/general_belief_replay_hard20.", ha="left", fontsize=13, color="#475569")

    panels = [
        (axes[0, 0], recon_mse, "Reconstruction MSE", "lower is better", "#2563eb"),
        (axes[0, 1], recon_r2, "Reconstruction R²", "higher is better", "#287d52"),
        (axes[1, 0], kl, "Consistency KL to target", "lower is better", "#7c3aed"),
        (axes[1, 1], ed, "Expected-distance difference (A)", "lower is better", "#c9452a"),
    ]
    for ax, y, title, subtitle, color in panels:
        ax.plot(layers, y, lw=2.4, color=color)
        ax.scatter(layers, y, s=30, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_title(title, loc="left", fontsize=14, pad=8)
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color="#64748b")
        ax.grid(True, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(1, 46)
        ax.set_xlabel("ESMFold block")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 1].set_ylabel("R²")
    axes[1, 0].set_ylabel("KL")
    axes[1, 1].set_ylabel("Å")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/general_belief_replay_hard20_metrics"))
    parser.add_argument("--out-prefix", type=Path, default=Path("runs/general_belief_replay_hard20_metrics/general_probe_training"))
    args = parser.parse_args()
    style()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"No metrics.csv files under {args.root}")
    write_summary(rows, args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"))
    plot(rows, args.out_prefix.with_name(args.out_prefix.name + "_metrics_by_block.png"))
    print(f"N={len(rows)}")
    print(f"recon_mse min={min(r['recon_mse'] for r in rows):.4f} max={max(r['recon_mse'] for r in rows):.4f}")
    print(f"recon_r2  min={min(r['recon_r2'] for r in rows):.4f} max={max(r['recon_r2'] for r in rows):.4f}")


if __name__ == "__main__":
    main()
