#!/usr/bin/env python3
"""Recompute the energy-descent vs geometric-convergence decoupling over ALL folds,
fixing (a) the silent name-mismatch that dropped every synthetic/hard fold and
(b) the TM>0.7 gate that excluded the wrong folds the rebuttal actually needs.

Within each protein, per layer we have:
    corr_vs_final[t]  -- correlation of layer-t distogram to the model's OWN final
                          (layer-47) distogram  = convergence to the answer it produces
    E[t]              -- LJ energy of the expected shape (per residue)
Per-step rates:  d_geo = Δ corr_vs_final ,  d_en = -Δ E  (energy descent rate).
Within-protein Pearson r(d_geo, d_en). If descent were just convergence, r->1.
"""
from pathlib import Path
import csv as _csv
import numpy as np

R = Path.home() / "recurrent-protein-folding" / "runs"
OUT = Path.home() / "recurrent-protein-folding" / "paper_figs"
OUT.mkdir(exist_ok=True)


def read_csv(path):
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
    return (xm * ym).sum() / denom if denom > 0 else 0.0


def cc_name(npz_name):
    """Map npz protein name -> contact_convergence.csv protein name."""
    n = npz_name
    for pre in ("benchmark_", "hard_synthetic_", "synthetic_"):
        if n.startswith(pre):
            n = n[len(pre):]
            break
    return n


d = np.load(R / "lj_sharpness_check.npz", allow_pickle=True)
names = [str(n) for n in d["names"]]
E = d["lj_of_mean"]; tms = np.asarray(d["tms"], float)
cc = read_csv(R / "contact_convergence.csv")
cc_proteins = set(cc["protein"].tolist())

rows = []  # (name, tm, r, n_steps)
for i, nm in enumerate(names):
    key = cc_name(nm)
    if key not in cc_proteins:
        print("  [no convergence data]", nm, "->", key)
        continue
    sel = cc["protein"] == key
    layer = cc["layer"][sel]; corr_all = cc["corr_vs_final"][sel]
    corr = corr_all[np.argsort(layer)]
    n = min(len(corr), E.shape[1])
    d_geo = np.diff(corr[:n])
    d_en = -np.diff(E[i, :n])
    m = np.isfinite(d_geo) & np.isfinite(d_en)
    if m.sum() > 5 and d_geo[m].std() > 0 and d_en[m].std() > 0:
        r = pearsonr(d_geo[m], d_en[m])
        rows.append((nm, tms[i], r, int(m.sum())))

rows.sort(key=lambda x: x[1])
allr = np.array([r for _, _, r, _ in rows])
tmr = np.array([t for _, t, _, _ in rows])

print(f"\n{'protein':32s} {'TM':>6s} {'r(descent,converge)':>20s}")
for nm, tm, r, ns in rows:
    print(f"{nm:32s} {tm:6.3f} {r:+20.3f}")

def band(lo, hi, label):
    msk = (tmr > lo) & (tmr <= hi)
    if msk.sum() == 0:
        print(f"{label:28s}  n=0"); return
    rr = allr[msk]
    print(f"{label:28s}  n={msk.sum():2d}  median r={np.median(rr):+.3f}  "
          f"mean={rr.mean():+.3f}  [{rr.min():+.3f},{rr.max():+.3f}]")

print("\n=== decoupling by fold-correctness band ===")
band(-1, 2.0, "ALL folds")
band(0.7, 2.0, "well-folded (TM>0.7)")
band(-1, 0.7, "wrong/marginal (TM<=0.7)")
band(-1, 0.4, "confidently wrong (TM<=0.4)")

# fraction with |r| small
print(f"\nfraction |r|<0.2 over ALL: {(np.abs(allr)<0.2).mean():.2f}  (n={len(allr)})")
print(f"median |r| ALL: {np.median(np.abs(allr)):.3f}")

# ---- figure: histogram split by correctness ----
try:
    import sys
    sys.path.insert(0, str(Path.home() / "recurrent-protein-folding" / "scripts"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import paper_style as ps
    ps.apply()
    mpl.rcParams.update({"savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})

    well = allr[tmr > 0.7]; wrong = allr[tmr <= 0.7]
    med_all = np.median(allr)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bins = np.linspace(-0.65, 0.65, 18)
    ax.hist(well, bins=bins, color=ps.PALETTE["blue"], alpha=0.80,
            edgecolor="white", label=f"well-folded (TM>0.7), n={len(well)}")
    ax.hist(wrong, bins=bins, color=ps.CHORD, alpha=0.65,
            edgecolor="white", label=f"wrong fold (TM$\\leq$0.7), n={len(wrong)}")
    ax.axvline(0, color=ps.SUBINK, lw=1.0, ls="--")
    ax.axvline(med_all, color=ps.INK, lw=2.2,
               label=f"median (all {len(allr)}) $r={med_all:+.3f}$")
    ax.set_xlim(-0.7, 0.7)
    ax.set_xlabel("within-protein correlation of per-step rates:\n"
                  r"energy descent ($-\Delta E$)  vs  geometric convergence ($\Delta$ corr-to-final)")
    ax.set_ylabel("number of proteins"); ps.finish(ax)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    ax.text(0.02, 0.62, "centered on 0:\ndescending energy $\\neq$\napproaching the answer",
            transform=ax.transAxes, fontsize=9, color=ps.SUBINK)
    fig.suptitle("Energy descent is decoupled from convergence to the answer",
                 x=0.5, fontsize=13.2, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "convergence_decoupling.pdf")
    fig.savefig(OUT / "convergence_decoupling.png", dpi=200)
    print("\nwrote convergence_decoupling.pdf/.png")
except Exception as e:
    print("FIG ERROR:", repr(e))
