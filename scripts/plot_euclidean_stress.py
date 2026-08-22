#!/usr/bin/env python3
"""Plot MDS stress (Euclidean-ness) of probe-predicted distance matrices vs layer.

stress_rmse = sqrt(mean((D_embedded - D_predicted)^2)) over all pairs.
Low stress = predicted distances are approximately Euclidean.

No cross-pair term in the probe loss (KL + recon_mse, both per-pair).
Euclidean structure is entirely emergent — evidence of probe faithfulness.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parents[1]
SIM_DIR  = ROOT / "runs/general_probe_simulations"
PROTEINS = ["ubq", "8fmi", "8qad", "6wxo", "9r2b", "T1137s4"]
COLORS   = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2"]


def load_stress(name: str) -> tuple[np.ndarray, np.ndarray]:
    p = SIM_DIR / name / "mds_general" / "mds_embedding_metrics.csv"
    layers, stress = [], []
    for row in csv.DictReader(p.open()):
        layers.append(int(row["layer"]))
        stress.append(float(row["stress_rmse"]))
    return np.array(layers), np.array(stress)


def main() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.13)
    fig.suptitle(
        "Probe-predicted distance matrices are approximately Euclidean at every layer",
        x=0.09, y=0.96, ha="left", fontsize=14, fontweight="bold", color="#172033")
    fig.text(
        0.09, 0.855,
        "MDS stress = RMSE between probe E[d] and best 3D embedding.  "
        "Probe loss is per-pair KL + recon — zero cross-pair consistency term.  "
        "Euclidean structure is emergent, not optimized → probe faithfulness.",
        ha="left", fontsize=10, color="#475569")

    all_stress = []
    for name, color in zip(PROTEINS, COLORS):
        try:
            layers, stress = load_stress(name)
        except FileNotFoundError:
            print(f"  skipping {name} (no data)")
            continue
        ax.plot(layers, stress, color=color, lw=2.0, label=name, zorder=3)
        ax.scatter(layers, stress, color=color, s=22, zorder=4)
        all_stress.append(stress)
        print(f"  {name:10s}  mean_stress={stress.mean():.3f}  max={stress.max():.3f}")

    if all_stress:
        mean = np.mean(all_stress, axis=0)
        ax.plot(layers, mean, color="#0f172a", lw=2.8, ls="--",
                label="mean", zorder=5)

    ax.set_xlabel("ESMFold trunk layer", fontsize=12)
    ax.set_ylabel("MDS stress (Å)", fontsize=12)
    ax.set_xlim(0.5, max(layers) + 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    out = ROOT / "runs/euclidean_stress_by_layer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
