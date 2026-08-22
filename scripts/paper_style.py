"""Shared figure aesthetic for the paper. Import and call `apply()` before plotting,
use `PALETTE` / `SEQ` for consistent colors, and `finish(ax)` to clean spines.

One place controls the look of every figure so the whole paper is visually consistent.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── palette ──────────────────────────────────────────────────────────────────
INK      = "#172033"   # near-black for titles
SUBINK   = "#475569"   # slate for secondary text
GRID     = "#e2e8f0"
EDGE     = "#cbd5e1"

PALETTE = {
    "blue":   "#2563eb",
    "green":  "#16a34a",
    "red":    "#dc2626",
    "amber":  "#d97706",
    "violet": "#7c3aed",
    "teal":   "#0d9488",
    "slate":  "#64748b",
    "rose":   "#e11d48",
}
# ordered sequence for multi-series plots
SEQ = [PALETTE[k] for k in ("blue", "green", "red", "amber", "violet", "teal", "rose", "slate")]

# semantic roles used across the paper
TRUNK   = PALETTE["green"]    # the trunk's own path / probe
CHORD   = PALETTE["red"]      # the straight chord / the model head (the "bad" comparison)
ENERGY  = PALETTE["blue"]
GOOD    = PALETTE["green"]
BAD     = PALETTE["red"]


def apply():
    mpl.rcParams.update({
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    EDGE,
        "axes.linewidth":    1.0,
        "axes.labelcolor":   INK,
        "axes.titlecolor":   INK,
        "axes.titleweight":  "bold",
        "axes.titlesize":    13,
        "axes.labelsize":    11.5,
        "axes.grid":         True,
        "grid.color":        GRID,
        "grid.linewidth":    0.7,
        "xtick.color":       SUBINK,
        "ytick.color":       SUBINK,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.frameon":    False,
        "legend.fontsize":   9.5,
        "font.family":       "DejaVu Sans",
        "figure.dpi":        110,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })


def finish(ax):
    """Hide top/right spines for a clean look."""
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def title_block(fig, title, subtitle=None, x=0.5, ha="center"):
    """Consistent left- or center-aligned title + subtitle block."""
    fig.suptitle(title, x=x, ha=ha, fontsize=15, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, 0.90, subtitle, ha=ha, fontsize=10, color=SUBINK)
