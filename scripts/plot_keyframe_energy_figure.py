#!/usr/bin/env python3
"""Compose the headline figure: 6 keyframes of the MDS-embedded fold + LJ energy curve."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GIF  = ROOT / "runs/general_probe_simulations/8fmi/mds_fixed_camera/distogram_mds_structure.gif"
ENG  = Path("/tmp/8fmi_energies.json")
OUT  = ROOT / "runs/headline_keyframe_energy.png"

LAYERS = [1, 10, 20, 30, 40, 46]


def crop_to_structure(img: Image.Image) -> Image.Image:
    """Crop out the title/footer; keep the protein structure region."""
    w, h = img.size
    # Heuristic crop: middle 78% horizontal, vertical 12%–88%
    return img.crop((int(w*0.18), int(h*0.10), int(w*0.82), int(h*0.92)))


def main():
    gif = Image.open(GIF)
    n_frames = gif.n_frames
    energies = json.loads(ENG.read_text())

    frames = []
    for L in LAYERS:
        f = min(n_frames - 1, max(0, (L - 1) * 3))
        gif.seek(f)
        frames.append(crop_to_structure(gif.convert("RGB")))

    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })

    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(2, len(LAYERS), height_ratios=[2.3, 1.0],
                           left=0.04, right=0.99, top=0.88, bottom=0.10,
                           wspace=0.04, hspace=0.40)

    fig.suptitle("ESMFold folds a protein by descending an energy landscape it was never asked to descend",
                 x=0.04, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color="#172033")
    fig.text(0.04, 0.91,
             "MDS embedding of the probe-read expected shape at six trunk layers (top).  "
             "LJ energy / residue of that shape across all 46 layers (bottom).  "
             "Probe loss has no Euclidean term and no cross-pair term; energy descent is emergent.",
             ha="left", fontsize=10, color="#475569")

    # Top row: structure frames
    for i, (L, img) in enumerate(zip(LAYERS, frames)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Layer {L}\nE/L = {energies[L-1]:+.2f}", fontsize=11)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1"); spine.set_linewidth(1.0)

    # Bottom: energy curve spanning full width
    ax_e = fig.add_subplot(gs[1, :])
    layers_x = np.arange(1, 47)
    e_arr    = np.array(energies)
    ax_e.plot(layers_x, e_arr, color="#0f172a", lw=2.4, zorder=3)
    ax_e.fill_between(layers_x, e_arr, e_arr.min() - 0.3,
                      color="#0f172a", alpha=0.08, zorder=2)
    for L in LAYERS:
        ax_e.axvline(L, color="#dc2626", lw=1.1, ls="--", alpha=0.55, zorder=1)
        ax_e.scatter([L], [energies[L-1]], color="#dc2626",
                     s=70, zorder=4, edgecolor="white", lw=1.2)
    ax_e.set_xlabel("ESMFold trunk layer", fontsize=11)
    ax_e.set_ylabel("LJ energy / residue", fontsize=11)
    ax_e.set_xlim(0.5, 46.5)
    ax_e.set_ylim(e_arr.min() - 0.3, max(e_arr.max() + 0.2, 0.0))
    ax_e.grid(True, lw=0.5)
    ax_e.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
