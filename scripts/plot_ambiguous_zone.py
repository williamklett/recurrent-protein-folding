#!/usr/bin/env python3
"""Show that pLDDT's signal vanishes in the ambiguous middle.

Panel row 1: scatter of each metric vs TM-score (all proteins).
Panel row 2: same scatter but restricted to proteins where pLDDT is in
             the "ambiguous zone" (middle 50% of pLDDT values, IQR).
             These are the cases no one can trivially call by eye.

In row 1 pLDDT looks decent (extremes do the work).
In row 2 pLDDT is a flat cloud; our metrics retain real signal.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

CSV_PATH  = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native/general_convergence_n40_native_logmae_summary.csv"
RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
OUT_PATH  = Path(__file__).resolve().parent.parent / \
    "runs/general_convergence_n40_native/ambiguous_zone.png"


def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def kabsch_align(p, q):
    pc = p - p.mean(0); qc = q - q.mean(0)
    h = pc.T @ qc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1, 1, det]) @ u.T).T + q.mean(0)


def tm_score(pred_ca, gt_ca):
    n = min(len(pred_ca), len(gt_ca))
    if n < 3: return float("nan")
    p = pred_ca[:n]; g = gt_ca[:n]
    aligned = kabsch_align(p, g)
    dists = np.sqrt(((aligned - g) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))


def get_tm(name, runs_root):
    for pat in ("benchmark_{n}", "hard_synthetic_miss_{n}", "synthetic_design_{n}", "{n}"):
        d = runs_root / pat.format(n=name)
        pred_p = d / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_p   = d / "gt_ca_distances.npz"
        if pred_p.exists() and gt_p.exists():
            return tm_score(
                np.load(pred_p)["pred_ca"].astype(float),
                np.load(gt_p)["ca_coords"].astype(float),
            )
    return float("nan")


def pearson(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def scatter_panel(ax, x, y, names, r_val, x_label, title,
                  color="#2563eb", highlight_mask=None):
    # Faded points outside mask, full colour inside
    if highlight_mask is None:
        highlight_mask = np.ones(len(x), dtype=bool)

    ax.scatter(x[~highlight_mask], y[~highlight_mask],
               color="#94a3b8", s=70, alpha=0.35, edgecolors="none", zorder=2)
    ax.scatter(x[highlight_mask], y[highlight_mask],
               color=color, s=110, alpha=0.88, edgecolors="#172033", lw=0.5, zorder=3)

    for i, nm in enumerate(names):
        if highlight_mask[i]:
            ax.annotate(nm, (x[i], y[i]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", color="#1e293b")

    ax.axhline(0.5, color="#0f172a", lw=0.7, ls="--", alpha=0.4)
    r_color = "#15803d" if abs(r_val) >= 0.5 else ("#d97706" if abs(r_val) >= 0.3 else "#dc2626")
    ax.text(0.04, 0.96, f"r = {r_val:+.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#a16207", alpha=0.95))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("TM-score vs PDB", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, lw=0.5)


def main():
    style()

    rows = list(csv.DictReader(CSV_PATH.open()))
    for r in rows:
        for k in ("mae", "rmsd", "plddt", "min_last5", "late_amp"):
            r[k] = float(r[k])
        r["tm"] = get_tm(r["name"], RUNS_ROOT)

    rows = [r for r in rows if not np.isnan(r["tm"])]
    print(f"n = {len(rows)} (with TM-score)")

    plddt   = np.array([r["plddt"]     for r in rows])
    ml5     = np.array([r["min_last5"] for r in rows])
    la      = np.array([r["late_amp"]  for r in rows])
    tm      = np.array([r["tm"]        for r in rows])
    names   = [r["name"] for r in rows]

    # Ambiguous zone = middle 50% of pLDDT (IQR)
    q25, q75 = np.percentile(plddt, 25), np.percentile(plddt, 75)
    ambig = (plddt >= q25) & (plddt <= q75)
    print(f"Ambiguous pLDDT zone: [{q25:.3f}, {q75:.3f}]  →  {ambig.sum()} proteins")

    # Full correlations
    r_plddt_full = pearson(plddt, tm)
    r_ml5_full   = pearson(-ml5, tm)   # lower min_last5 = better, so negate for +ve correlation
    r_la_full    = pearson(la, tm)

    # Ambiguous-zone correlations
    r_plddt_amb  = pearson(plddt[ambig], tm[ambig])
    r_ml5_amb    = pearson(-ml5[ambig], tm[ambig])
    r_la_amb     = pearson(la[ambig],   tm[ambig])

    print(f"\nFull dataset (n={len(rows)}):")
    print(f"  pLDDT    r = {r_plddt_full:+.3f}")
    print(f"  min_last5 r = {r_ml5_full:+.3f}  (negated so +ve = better)")
    print(f"  late_amp  r = {r_la_full:+.3f}")
    print(f"\nAmbiguous zone only (pLDDT ∈ [{q25:.2f}, {q75:.2f}], n={ambig.sum()}):")
    print(f"  pLDDT    r = {r_plddt_amb:+.3f}")
    print(f"  min_last5 r = {r_ml5_amb:+.3f}")
    print(f"  late_amp  r = {r_la_amb:+.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.07,
                        wspace=0.27, hspace=0.38)

    fig.suptitle(
        "pLDDT's signal evaporates in the ambiguous middle — convergence metrics don't",
        x=0.055, y=0.96, ha="left", fontsize=17, fontweight="bold", color="#172033",
    )
    band_str = f"pLDDT IQR = [{q25:.2f}, {q75:.2f}]"
    fig.text(
        0.055, 0.91,
        f"Top row: all {len(rows)} proteins.  "
        f"Bottom row: {ambig.sum()} proteins in the ambiguous pLDDT zone ({band_str}) — "
        "the cases no one can trivially call by eye.",
        ha="left", fontsize=11, color="#475569",
    )

    all_mask = np.ones(len(rows), dtype=bool)

    # Row 0 — full dataset
    scatter_panel(axes[0, 0], plddt, tm, names, r_plddt_full,
                  "pLDDT", f"All proteins — pLDDT  (r={r_plddt_full:+.3f})",
                  color="#2563eb", highlight_mask=all_mask)
    scatter_panel(axes[0, 1], -ml5, tm, names, r_ml5_full,
                  "−min_last5  (higher = more sharpening)", f"All proteins — min_last5  (r={r_ml5_full:+.3f})",
                  color="#16a34a", highlight_mask=all_mask)
    scatter_panel(axes[0, 2], la, tm, names, r_la_full,
                  "late_amp  (higher = more sharpening)", f"All proteins — late_amp  (r={r_la_full:+.3f})",
                  color="#dc2626", highlight_mask=all_mask)

    # Row 1 — ambiguous zone only
    scatter_panel(axes[1, 0], plddt[ambig], tm[ambig], [n for n, m in zip(names, ambig) if m],
                  r_plddt_amb,
                  "pLDDT", f"Ambiguous zone — pLDDT  (r={r_plddt_amb:+.3f})",
                  color="#2563eb", highlight_mask=np.ones(ambig.sum(), dtype=bool))
    scatter_panel(axes[1, 1], -ml5[ambig], tm[ambig], [n for n, m in zip(names, ambig) if m],
                  r_ml5_amb,
                  "−min_last5", f"Ambiguous zone — min_last5  (r={r_ml5_amb:+.3f})",
                  color="#16a34a", highlight_mask=np.ones(ambig.sum(), dtype=bool))
    scatter_panel(axes[1, 2], la[ambig], tm[ambig], [n for n, m in zip(names, ambig) if m],
                  r_la_amb,
                  "late_amp", f"Ambiguous zone — late_amp  (r={r_la_amb:+.3f})",
                  color="#dc2626", highlight_mask=np.ones(ambig.sum(), dtype=bool))

    # Shade ambiguous band on top row pLDDT scatter
    for ax in axes[0]:
        ax.axvspan(q25, q75, alpha=0.08, color="#f59e0b", zorder=1,
                   label=f"ambiguous band [{q25:.2f}–{q75:.2f}]")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
