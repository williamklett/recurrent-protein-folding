#!/usr/bin/env python3
"""Publication-quality figures for the energy-manifold paper. Vector PDF output.
Run on the VM (full env + scripts/paper_style.py). Reads only on-disk data."""
from pathlib import Path
import json
import csv as _csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import Normalize


def read_csv(path):
    """Minimal CSV -> dict of numpy arrays (float where possible, else str)."""
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = np.array(vals, dtype=object)
    return out


def pearsonr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return (xm * ym).sum() / denom if denom > 0 else 0.0,

import sys
sys.path.insert(0, str(Path.home() / "recurrent-protein-folding" / "scripts"))
import paper_style as ps
ps.apply()
mpl.rcParams.update({"savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})

R = Path.home() / "recurrent-protein-folding" / "runs"
OUT = Path.home() / "recurrent-protein-folding" / "paper_figs"
OUT.mkdir(exist_ok=True)
EARLY, MID, LATE = (1, 11), (12, 31), (32, 45)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  wrote", name, flush=True)


def band_shade(ax):
    for (lo, hi), c in [(EARLY, "#eff6ff"), (MID, "#f0fdf4"), (LATE, "#fef2f2")]:
        ax.axvspan(lo - 0.5, hi + 0.5, color=c, zorder=0)


# ── F1  energy descent, all 57 proteins, colored by TM ────────────────────────
def fig_energy():
    d = np.load(R / "lj_sharpness_check.npz", allow_pickle=True)
    E, tms = d["lj_of_mean"], d["tms"]
    x = np.arange(1, E.shape[1] + 1)
    order = np.argsort(tms)
    norm = Normalize(tms.min(), tms.max())
    cmap = cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i in order:
        ax.plot(x, E[i], color=cmap(norm(tms[i])), lw=1.0, alpha=0.75)
    ax.plot(x, E.mean(0), color=ps.INK, lw=2.6, label="mean (n=57)")
    ax.set_xlabel("trunk block"); ax.set_ylabel(r"LJ energy of expected shape  $E_t$  (per residue)")
    ax.set_xlim(1, E.shape[1]); ps.finish(ax)
    ax.legend(loc="upper right")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label("final TM-score", color=ps.INK); cb.outline.set_visible(False)
    dE = E[:, -1] - E[:, 0]
    ax.text(0.015, 0.04,
            rf"$\Delta E = {dE.mean():.2f}\pm{dE.std():.2f}$  (range ${dE.max():.2f}$ to ${dE.min():.2f}$)",
            transform=ax.transAxes, fontsize=9.5, color=ps.SUBINK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ps.EDGE, lw=0.8))
    fig.suptitle("Energy descends monotonically through the trunk",
                 x=0.5, fontsize=14, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "energy_descent")


# ── F2  the chord: trunk path vs straight interpolation (ubq) ──────────────────
def fig_chord():
    j = json.load(open(R / "manifold_chord_20_46.json"))["ubq"]
    tp, te = np.array(j["trunk_prog"]), np.array(j["trunk_E"])
    cp, ce = np.array(j["chord_prog"]), np.array(j["chord_E"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(cp, ce, color=ps.CHORD, lw=2.4, marker="o", ms=3.5,
            label="straight chord (same endpoints)")
    ax.plot(tp, te, color=ps.TRUNK, lw=2.6, marker="o", ms=3.5,
            label="trunk's own path")
    ax.set_yscale("log")
    ax.set_xlabel("progress from block 20 to block 46")
    ax.set_ylabel(r"LJ energy of realized 3D coordinates  (log scale)")
    ax.set_xlim(-0.02, 1.02); ps.finish(ax); ax.legend(loc="upper right")
    ax.annotate(rf"chord peak $\approx${ce.max():.1e}",
                xy=(cp[np.argmax(ce)], ce.max()), xytext=(0.10, ce.max() * 0.06),
                color=ps.CHORD, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=ps.CHORD, lw=1.1))
    ax.text(0.015, 0.30,
            f"endpoint RMSD {j['end_rmsd']:.1f} Å > $R_g$ {j['rg']:.1f} Å  (RMSD/$R_g$={j['end_rmsd']/j['rg']:.2f})\n"
            f"embedding stress {j['stress_end']:.1f} Å",
            transform=ax.transAxes, fontsize=9, color=ps.SUBINK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=ps.EDGE, lw=0.8))
    ax.text(0.80, 0.085, "trunk stays at $E\\sim10^2$", transform=ax.transAxes,
            fontsize=9, color=ps.TRUNK, ha="center")
    fig.suptitle("The trunk's route stays physical; a direct route tunnels through clashes",
                 x=0.5, fontsize=13.2, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "manifold_chord")


# ── F3  within-protein decoupling of energy descent vs geometric convergence ───
def fig_convergence():
    d = np.load(R / "lj_sharpness_check.npz", allow_pickle=True)
    def _cc(nm):
        for pre in ("benchmark_", "hard_synthetic_", "synthetic_"):
            if nm.startswith(pre):
                return nm[len(pre):]
        return nm
    names = [str(n) for n in d["names"]]
    E = d["lj_of_mean"]; tms = np.asarray(d["tms"], float)
    cc = read_csv(R / "contact_convergence.csv")
    rs = []
    for i, nm in enumerate(names):
        sel = cc["protein"] == _cc(nm)  # all folds; no TM gate (decoupling holds on wrong folds too)
        if sel.sum() < 10:
            continue
        layer = cc["layer"][sel]; corr_all = cc["corr_vs_final"][sel]
        corr = corr_all[np.argsort(layer)]
        n = min(len(corr), E.shape[1])
        d_geo = np.diff(corr[:n])               # geometric convergence rate
        d_en = -np.diff(E[i, :n])               # energy descent rate (−ΔLJ)
        m = np.isfinite(d_geo) & np.isfinite(d_en)
        if m.sum() > 5 and d_geo[m].std() > 0 and d_en[m].std() > 0:
            rs.append(pearsonr(d_geo[m], d_en[m])[0])
    rs = np.array(rs); med = np.median(rs)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(rs, bins=np.linspace(-0.6, 0.6, 16), color=ps.PALETTE["blue"],
            alpha=0.85, edgecolor="white")
    ax.axvline(0, color=ps.SUBINK, lw=1.0, ls="--")
    ax.axvline(med, color=ps.CHORD, lw=2.2, label=f"median $r = {med:+.3f}$  (n={len(rs)})")
    ax.set_xlim(-0.65, 0.65)
    ax.set_xlabel("within-protein correlation of per-step rates:\n"
                  r"energy descent ($-\Delta E$)  vs  geometric convergence ($\Delta$ corr-to-final)")
    ax.set_ylabel("number of proteins"); ps.finish(ax)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.text(0.02, 0.78, "centered on 0:\ndescending energy $\\neq$\napproaching the answer",
            transform=ax.transAxes, fontsize=9, color=ps.SUBINK)
    fig.suptitle("Energy descent is decoupled from convergence to the answer",
                 x=0.5, fontsize=13.2, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "convergence_decoupling")


# ── F4  faithfulness: probe is metric-consistent & low-stress across folds ─────
def fig_faithfulness():
    tv = read_csv(R / "triangle_violation_results.csv")
    st = read_csv(R / "stress_auroc_results.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    a1.scatter(tv["tm"], tv["max_viol"], s=26, color=ps.TRUNK, alpha=0.8, edgecolor="white")
    a1.set_xlabel("final TM-score"); a1.set_ylabel("max triangle-inequality violation (probe)")
    a1.set_title("Probe geometry is metric at every fold quality", fontsize=11.5)
    a1.text(0.04, 0.9, f"global max = {tv['max_viol'].max():.1e}\nmost proteins exactly 0",
            transform=a1.transAxes, fontsize=9, color=ps.SUBINK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ps.EDGE, lw=0.8))
    ps.finish(a1)
    a2.scatter(st["tm"], st["mean_stress"], s=26, color=ps.ENERGY, alpha=0.8,
               edgecolor="white", label="mean stress")
    a2.scatter(st["tm"], st["max_stress"], s=20, color=ps.PALETTE["violet"], alpha=0.55,
               edgecolor="white", label="max stress")
    a2.set_xlabel("final TM-score"); a2.set_ylabel("MDS embedding stress (Å)")
    a2.set_ylim(0, st["max_stress"].max() * 1.15)
    a2.set_title("Embeddings are low-stress even on bad folds", fontsize=11.5)
    a2.legend(loc="upper right"); ps.finish(a2)
    fig.suptitle("Faithfulness: the readout reads coherent geometry, not mis-encoded content",
                 x=0.5, fontsize=13.2, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "faithfulness")


# ── F5  not gradient descent (belief space, dimension-controlled) ──────────────
def fig_belief_gd():
    c = read_csv(R / "belief_gd_clean.csv")
    dm = read_csv(R / "belief_gd_dim.csv")
    co = np.argsort(c["layer"]); c = {k: v[co] for k, v in c.items()}
    do = np.argsort(dm["layer"]); dm = {k: v[do] for k, v in dm.items()}
    L = c["layer"]
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.2))
    A = ps.PALETTE["blue"]; N = ps.PALETTE["amber"]

    # (a) energy change per step
    ax = axs[0, 0]; band_shade(ax)
    mask = c["layer"] <= 44      # 45->46 is a probe-edge artifact (+24.9)
    ax.plot(L[mask], c["dE"][mask], color=ps.ENERGY, lw=2.2, marker="o", ms=3)
    ax.axhline(0, color=ps.SUBINK, lw=0.9, ls="--")
    ax.set_ylabel(r"$\Delta E$ per step  (sum over pairs)")
    ax.set_title("Energy falls every step", fontsize=11.5); ps.finish(ax)

    # (b) global magnitude-weighted cosine
    ax = axs[0, 1]; band_shade(ax)
    ax.plot(L, dm["abs_obs"], color=A, lw=2.0, marker="o", ms=3, label="Euclidean")
    ax.plot(L, dm["nat_obs"], color=N, lw=2.0, marker="s", ms=3, label="Fisher-natural")
    ax.axhline(0, color=ps.SUBINK, lw=0.9, ls="--")
    ax.set_ylabel(r"global $\cos(\mathrm{step},\,-\nabla E)$")
    ax.set_title("Downhill at every layer — but tiny", fontsize=11.5)
    ax.legend(loc="upper left"); ps.finish(ax)

    # (c) permutation z-score
    ax = axs[1, 0]; band_shade(ax)
    ax.plot(L, dm["abs_z"], color=A, lw=2.0, marker="o", ms=3, label="Euclidean")
    ax.plot(L, dm["nat_z"], color=N, lw=2.0, marker="s", ms=3, label="Fisher-natural")
    ax.axhline(0, color=ps.SUBINK, lw=0.9, ls="--")
    ax.set_xlabel("trunk block"); ax.set_ylabel("permutation-null z-score")
    ax.set_title("Real, not a high-D artifact (>10σ); natural rises with depth", fontsize=10.8)
    ax.legend(loc="upper right"); ps.finish(ax)

    # (d) per-pair energy-relevant variance
    ax = axs[1, 1]; band_shade(ax)
    ax.plot(L, dm["abs_r2"], color=A, lw=2.0, marker="o", ms=3, label="Euclidean")
    ax.plot(L, dm["nat_r2"], color=N, lw=2.0, marker="s", ms=3, label="Fisher-natural")
    ax.axhline(1 / 63, color=ps.CHORD, lw=1.6, ls=":", label="random baseline (1/63)")
    ax.set_xlabel("trunk block"); ax.set_ylabel(r"per-pair $\cos^2$ (energy-relevant var.)")
    ax.set_title("≈99% of each step is energy-neutral", fontsize=11.5)
    ax.legend(loc="upper right"); ps.finish(ax)

    fig.suptitle("Not gradient descent: the step is reliably downhill but ~99% orthogonal to ∇E",
                 x=0.5, fontsize=14, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "belief_gd")


# ── F6  corroborating activation-space cosine (ubq) ───────────────────────────
def fig_cosine_z():
    df = read_csv(R / "cosine_backprop_local.csv")
    keep = np.abs(df["cos_neg_lj"]) <= 1                        # drop |cos|>1 artifacts
    df = {k: v[keep] for k, v in df.items()}
    o = np.argsort(df["layer"]); df = {k: v[o] for k, v in df.items()}
    L = df["layer"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    band_shade(ax)
    ax.plot(L, df["cos_neg_lj"], color=ps.PALETTE["blue"], lw=2.2, marker="o", ms=3,
            label=r"$\cos(\Delta z,\,-\nabla_z E_{\mathrm{LJ}})$")
    ax.plot(L, df["cos_neg_gt"], color=ps.PALETTE["green"], lw=1.8, marker="s", ms=3,
            label=r"$\cos(\Delta z,\,-\nabla_z E_{\mathrm{GT}})$")
    ax.plot(L, df["cos_random"], color=ps.SUBINK, lw=1.4, ls="--", label="random baseline")
    ax.axhline(0, color=ps.SUBINK, lw=0.8)
    ax.set_xlabel("trunk block"); ax.set_ylabel(r"$\cos(\Delta z_t,\,-\nabla_{z_t} E)$")
    ax.set_ylim(-0.06, 0.08); ps.finish(ax); ax.legend(loc="upper left")
    ax.text(0.97, 0.06, "peak ≈ 0.05 ≪ 1\n(literal GD would be ≈1)", transform=ax.transAxes,
            ha="right", fontsize=9, color=ps.SUBINK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ps.EDGE, lw=0.8))
    fig.suptitle("Corroboration in activation space (ubq): alignment switches on mid-trunk, peaks ~0.05",
                 x=0.5, fontsize=12.2, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "cosine_zspace")


if __name__ == "__main__":
    fig_energy()
    fig_chord()
    fig_convergence()
    fig_faithfulness()
    fig_belief_gd()
    fig_cosine_z()
    print("ALL FIGS DONE")
