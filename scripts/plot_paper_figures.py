#!/usr/bin/env python3
"""Plot paper figures from data ALREADY CACHED in runs/ (no recompute).

Sources (all pre-existing on disk):
  runs/lj_sharpness_check.npz      names, tms, lj_of_mean (57x46)  -> per-layer LJ energy
  runs/contact_convergence.csv     per protein/layer corr_vs_final  -> geometric convergence
  runs/cosine_lj.csv               per protein/layer cos(dz,-grad)  -> energy-gradient alignment
  runs/manifold_chord_20_46.json   trunk vs straight-chord energy   -> ubq chord

Produces, into paper/figures/:
  lj_energy_grid.png         per-protein LJ-energy trajectories, colored by TM
  convergence_vs_energy.png  within-protein convergence rate vs energy-descent rate
  cosine_alignment.png       cos(dz,-grad E) vs cos(dz,-grad target) per layer
  manifold_chord.png         clean ubq-only chord vs trunk (log y)

stress_by_layer.png and keyframe_energy.png are kept as previously rendered
(their per-layer source is not cached; regenerating them would require recompute).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import paper_style as ps

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = ROOT / "paper/figures"
FIGS.mkdir(parents=True, exist_ok=True)
ps.apply()


def _norm(s: str) -> str:
    for p in ("benchmark_", "hard_synthetic_miss_", "synthetic_design_", "miss_"):
        s = s.replace(p, "")
    return s


def load_energy():
    """name -> per-layer LJ energy (len 46, layers 1..46); name -> TM."""
    d = np.load(RUNS / "lj_sharpness_check.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    lj = d["lj_of_mean"]            # (57, 46)
    tms = d["tms"]
    E = {_norm(n): lj[i] for i, n in enumerate(names)}
    T = {_norm(n): float(tms[i]) for i, n in enumerate(names)}
    return E, T


def load_convergence():
    """name -> per-layer corr_vs_final (geometric approach to the final shape)."""
    rows = list(csv.DictReader(open(RUNS / "contact_convergence.csv")))
    by = {}
    for r in rows:
        by.setdefault(_norm(r["protein"]), {})[int(r["layer"])] = float(r["corr_vs_final"])
    out = {}
    for p, dd in by.items():
        layers = sorted(dd)
        out[p] = np.array([dd[L] for L in layers])
    return out


# ── 1. energy grid (overlay colored by TM) ────────────────────────────────────
def fig_energy(E, T):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(top=0.86, right=0.88)
    cmap = cm.get_cmap("RdYlGn"); norm = Normalize(0.0, 1.0)
    for p in sorted(E, key=lambda p: T.get(p, 0)):
        lj = E[p]
        ax.plot(np.arange(1, len(lj) + 1), lj, color=cmap(norm(T.get(p, 0))),
                lw=1.3, alpha=0.85)
    ps.title_block(fig,
        "The trunk descends an energy landscape it was never trained on",
        "LJ energy / residue of the probe-read expected shape, per trunk layer "
        "(57 proteins).  Color = fold quality (TM).", x=0.5)
    ax.set_xlabel("ESMFold trunk layer")
    ax.set_ylabel("LJ energy / residue")
    ps.finish(ax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02); cb.set_label("TM-score (fold quality)", color=ps.SUBINK)
    fig.savefig(FIGS / "lj_energy_grid.png"); plt.close(fig)
    print("wrote lj_energy_grid.png")


# ── 2. within-protein convergence rate vs energy-descent rate ──────────────────
def fig_conv_energy(E, T):
    C = load_convergence()
    shared = sorted(set(E) & set(C))
    corrs, good = [], []
    for p in shared:
        tm = T.get(p, 0.0)
        if tm < 0.7:
            continue
        lj, c = E[p], C[p]
        n = min(len(lj), len(c))
        dEdesc = -np.diff(lj[:n])       # energy descent (positive = down)
        dconv = np.diff(c[:n])          # geometric approach toward final
        m = np.isfinite(dEdesc) & np.isfinite(dconv)
        if m.sum() < 5:
            continue
        r = np.corrcoef(dconv[m], dEdesc[m])[0, 1]
        corrs.append(r); good.append((p, tm, dconv[m], dEdesc[m], r))
    corrs = np.array(corrs)
    good.sort(key=lambda g: -g[1]); rep = good[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(top=0.84, wspace=0.28)
    ps.title_block(fig,
        "Energy descent and geometric convergence are different processes",
        "Within each protein, the layers of fastest convergence are not the layers "
        "of fastest energy descent.", x=0.5)

    p, tm, dc, dE, r = rep
    axes[0].scatter(dc, dE, s=26, color=ps.ENERGY, alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel(r"geometric convergence rate  $\Delta$ corr(d$_t$, d$_{final}$)")
    axes[0].set_ylabel(r"energy descent rate  $-\Delta$ LJ")
    axes[0].set_title(f"{p}  (TM={tm:.2f}),  r = {r:+.2f}", fontsize=11)
    ps.finish(axes[0])

    axes[1].axvline(0, color=ps.SUBINK, lw=0.9, ls=":")
    axes[1].hist(corrs, bins=10, color=ps.PALETTE["violet"], alpha=0.85, edgecolor="white")
    axes[1].axvline(np.median(corrs), color=ps.BAD, lw=2.2,
                    label=f"median r = {np.median(corrs):+.2f}  (n={len(corrs)})")
    axes[1].set_xlabel("within-protein correlation of the two rates")
    axes[1].set_ylabel("# proteins (TM > 0.7)")
    axes[1].set_title("Distribution across proteins", fontsize=11)
    axes[1].legend(loc="upper left")
    ps.finish(axes[1])

    fig.savefig(FIGS / "convergence_vs_energy.png"); plt.close(fig)
    print(f"wrote convergence_vs_energy.png  median within-protein r = {np.median(corrs):+.3f}  n={len(corrs)}")
    return float(np.median(corrs)), len(corrs)


# ── 3. cosine alignment with energy gradient (backprop, T+1 distogram readout) ──
def fig_cosine():
    """Correct metric: cos(dz_t, -grad_{z_t} E_LJ(distogram_{t+1})), one block of
    autograd. Source: runs/cosine_backprop_local.csv (per layer). Peaks ~0.05
    mid-trunk; nowhere near 1 -> not gradient descent."""
    src = RUNS / "cosine_backprop_local.csv"
    if not src.exists():
        print("no cosine_backprop_local.csv; skipping cosine fig"); return
    rows = list(csv.DictReader(open(src)))

    def series(col):
        L, v = [], []
        for r in rows:
            x = r.get(col, "")
            if x in ("", "nan"):
                continue
            f = float(x)
            if not np.isfinite(f) or abs(f) > 1:   # drop autograd artifact rows
                continue
            L.append(int(r["layer"])); v.append(f)
        order = np.argsort(L)
        return np.array(L)[order], np.array(v)[order]

    Llj, lj = series("cos_neg_lj")
    Lgt, gt = series("cos_neg_gt")
    Lrn, rnd = series("cos_random")
    peak = float(np.max(lj)); peak_L = int(Llj[np.argmax(lj)])
    mean_all = float(np.mean(lj))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.subplots_adjust(top=0.86)
    ax.axhline(0, color=ps.SUBINK, lw=0.9, ls=":")
    ax.plot(Lgt, gt,  color=ps.GOOD, lw=2.0, marker="o", ms=3, alpha=0.9,
            label=r"$-\nabla_{z_t}$ (distance to target)")
    ax.plot(Llj, lj,  color=ps.BAD,  lw=2.4, marker="o", ms=3,
            label=r"$-\nabla_{z_t} E_{\mathrm{LJ}}(\mathrm{distogram}_{t+1})$ (energy)")
    ax.plot(Lrn, rnd, color=ps.PALETTE["slate"], lw=1.4, ls="--", label="random direction")
    ax.annotate(f"peak {peak:.3f}\n(block {peak_L})",
                xy=(peak_L, peak), xytext=(peak_L + 4, peak + 0.012),
                fontsize=9, color=ps.BAD,
                arrowprops=dict(arrowstyle="->", color=ps.BAD, lw=1.0))
    ps.title_block(fig,
        "The trunk is not doing gradient descent on energy",
        r"Per-block $\cos(\Delta z_t,\,-\nabla_{z_t}\,\cdot\,)$ (ubiquitin, backprop to the "
        r"layer-$t{+}1$ distogram readout).  Energy alignment peaks $\sim$0.05 mid-trunk "
        r"$\,\ll 1$.", x=0.5)
    ax.set_xlabel("ESMFold trunk layer")
    ax.set_ylabel(r"cosine alignment of trunk update $\Delta z_t$")
    ax.legend(loc="lower right")
    ps.finish(ax)
    fig.savefig(FIGS / "cosine_alignment.png"); plt.close(fig)
    print(f"wrote cosine_alignment.png  (peak cos(dz,-grad E_LJ) = {peak:+.4f} @ block {peak_L}; all-layer mean = {mean_all:+.4f})")
    return peak


# ── 4. clean ubq chord (log y) ─────────────────────────────────────────────────
def fig_chord():
    src = RUNS / "manifold_chord_20_46.json"
    if not src.exists():
        print("no chord json; skipping chord fig"); return
    d = json.load(open(src))
    if "ubq" not in d:
        print("no ubq in chord json"); return
    r = d["ubq"]
    tp, te = np.array(r["trunk_prog"]), np.array(r["trunk_E"])
    cp, ce = np.array(r["chord_prog"]), np.array(r["chord_E"])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(top=0.84)
    ax.plot(cp, ce, color=ps.CHORD, lw=2.6, label="straight coordinate chord")
    ax.plot(tp, te, color=ps.TRUNK, lw=2.6, marker="o", ms=3, label="trunk's actual path")
    ax.scatter([0, 1], [te[0], te[-1]], color=ps.INK, s=60, zorder=5,
               label="shared on-manifold endpoints")
    ax.set_yscale("log")
    ps.title_block(fig,
        "The trunk's path stays low; the straight chord tunnels through clashes",
        "Ubiquitin, blocks 20->46 (endpoint RMSD 11.6 A > $R_g$).  Energy from "
        "realized 3D coordinates.  Log scale.", x=0.5)
    ax.set_xlabel("path progress  (block 20 -> 46)")
    ax.set_ylabel("LJ energy / residue  (from 3D coords, log)")
    ax.legend(loc="upper right")
    ps.finish(ax)
    fig.savefig(FIGS / "manifold_chord.png"); plt.close(fig)
    print("wrote manifold_chord.png")


def main():
    E, T = load_energy()
    print(f"loaded {len(E)} proteins with per-layer energy")
    fig_energy(E, T)
    fig_conv_energy(E, T)
    fig_cosine()
    fig_chord()


if __name__ == "__main__":
    main()
