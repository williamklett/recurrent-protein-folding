#!/usr/bin/env python3
"""Lennard-Jones energy from probe-predicted pairwise distances, per layer.

E = sum_{|i-j|>2} 4*eps * [ (sigma/d)^12 - (sigma/d)^6 ]

sigma = 6.8 Å  (typical Cα contact distance)
eps   = 1.0    (arbitrary units, we care about trends not absolute values)

If ESMFold is doing implicit energy minimization, E/L should decrease
(become more negative) monotonically through layers 1→46.

Bad folds should plateau at higher energy than good folds.

Output: lj_energy_results.csv  +  lj_energy_plot.png
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"

SIGMA      = 6.8   # Å
EPS        = 1.0
SEQ_SEP    = 3     # skip |i-j| <= this (bonded neighbours)
FIRST, LAST = 1, 46


def load_probe(layer, device):
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters(): p.requires_grad_(False)
    return pr


def get_ed(z_layer, probe, centers, device):
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    pairs = z_layer[ii_f, jj_f].to(device)
    ii_d, jj_d = ii_f.to(device), jj_f.to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_d, jj_d, L)
        ed = (F.softmax(logits, -1) * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy()


def lj_energy(D: np.ndarray, sigma: float = SIGMA, seq_sep: int = SEQ_SEP) -> float:
    """Per-residue LJ energy from upper-triangle of distance matrix."""
    L = D.shape[0]
    ii, jj = np.triu_indices(L, k=seq_sep + 1)
    d = D[ii, jj].clip(min=0.5)   # avoid singularity
    r = sigma / d
    e = 4 * EPS * (r**12 - r**6)
    return float(e.sum()) / L      # normalise by length


def kabsch_tm(pred, gt):
    n = min(len(pred), len(gt))
    p, g = pred[:n].copy(), gt[:n].copy()
    p -= p.mean(0); gc = g - g.mean(0)
    H = p.T @ gc; U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1,1,np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    pa = p @ R.T + g.mean(0)
    d  = np.sqrt(((pa - g)**2).sum(-1))
    d0 = max(0.5, 1.24*np.cbrt(max(n-15,1)) - 1.8)
    return float(np.mean(1/(1+(d/d0)**2)))


def auroc(scores, labels):
    n_pos, n_neg = labels.sum(), len(labels)-labels.sum()
    if n_pos == 0 or n_neg == 0: return float("nan")
    order = np.argsort(-scores)
    auc = tpr = fpr = prev = 0.0
    for i in order:
        if labels[i]: tpr += 1/n_pos
        else: fpr += 1/n_neg; auc += tpr*(fpr-prev); prev=fpr
    return float(auc)


def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32)

    proteins = []
    for act_p in sorted(RUNS_DIR.glob("*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p   = run_dir / "gt_ca_distances.npz"
        pred_p = next(run_dir.glob("**/esmfold_prediction_ca.npz"), None)
        if gt_p.exists() and pred_p:
            proteins.append((run_dir.name, act_p, gt_p, pred_p))

    print(f"Found {len(proteins)} proteins")
    n_layers = LAST - FIRST + 1
    rows     = []
    all_curves = []   # (name, tm, energy_per_layer)

    for pi, (name, act_p, gt_p, pred_p) in enumerate(proteins):
        print(f"[{pi+1}/{len(proteins)}] {name}", flush=True)
        act   = torch.load(act_p, map_location="cpu", weights_only=False)
        z_all = act["z"].squeeze(1).float()
        gt_ca   = np.load(gt_p)["ca_coords"].astype(float)
        pred_ca = np.load(pred_p)["pred_ca"].astype(float)
        tm = kabsch_tm(pred_ca, gt_ca)

        energies = []
        for t in range(FIRST, LAST + 1):
            probe = load_probe(t, device)
            D = get_ed(z_all[t], probe, centers, device)
            energies.append(lj_energy(D))
            del probe
        torch.cuda.empty_cache()

        energies = np.array(energies)
        final_e  = float(energies[-1])
        delta_e  = float(energies[-1] - energies[0])   # negative = decreased = good
        min_e    = float(energies.min())
        monotone = float(np.mean(np.diff(energies) <= 0))  # fraction decreasing steps

        print(f"  TM={tm:.3f}  E[1]={energies[0]:.2f}  E[46]={final_e:.2f}  "
              f"ΔE={delta_e:+.2f}  mono={monotone:.2f}")
        rows.append({"name": name, "tm": tm, "E_first": round(energies[0],3),
                     "E_final": round(final_e,3), "delta_E": round(delta_e,3),
                     "min_E": round(min_e,3), "monotone_frac": round(monotone,3)})
        all_curves.append((name, tm, energies))

    # AUROC (higher delta_E = less decrease = worse fold)
    tm_arr    = np.array([r["tm"]      for r in rows])
    delta_arr = np.array([r["delta_E"] for r in rows])
    final_arr = np.array([r["E_final"] for r in rows])
    fail      = tm_arr < 0.5

    auc_delta = auroc(delta_arr,  fail)   # less negative delta = failure
    auc_final = auroc(final_arr,  fail)   # higher final energy = failure

    print(f"\nN={len(rows)}  failures(TM<0.5)={fail.sum()}")
    print(f"AUROC delta_E={auc_delta:.4f}  (less energy drop = failure)")
    print(f"AUROC E_final={auc_final:.4f}  (higher final energy = failure)")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_csv = ROOT / "runs/lj_energy_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv}")

    # ── plot ──────────────────────────────────────────────────────────────────
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12, wspace=0.30)
    fig.suptitle("Implicit energy minimization in ESMFold trunk layers",
                 x=0.07, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color="#172033")
    fig.text(0.07, 0.88,
             f"LJ energy E/L from probe-predicted Cα distances at each layer.  "
             f"σ={SIGMA}Å, |i-j|>{SEQ_SEP}.  "
             f"AUROC: ΔE={auc_delta:.3f}  E_final={auc_final:.3f}  N={len(rows)}",
             ha="left", fontsize=10, color="#475569")

    # Panel 0: energy curves per protein, coloured by TM
    ax = axes[0]
    layers = np.arange(FIRST, LAST + 1)
    cmap   = plt.cm.RdYlGn
    for name, tm, energies in sorted(all_curves, key=lambda x: x[1]):
        color = cmap(tm)
        ax.plot(layers, energies, color=color, lw=1.2, alpha=0.7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="TM-score")
    ax.set_xlabel("ESMFold trunk layer", fontsize=11)
    ax.set_ylabel("LJ energy / residue (a.u.)", fontsize=11)
    ax.set_title("Energy trajectory per protein\n(green=good fold, red=bad fold)", fontsize=11)
    ax.grid(True, lw=0.5)

    # Panel 1: ΔE vs TM scatter
    ax2 = axes[1]
    colors2 = ["#ef4444" if f else "#16a34a" for f in fail]
    ax2.scatter(delta_arr, tm_arr, c=colors2, s=70,
                edgecolors="#172033", lw=0.6, zorder=3)
    ax2.axvline(0, color="#94a3b8", lw=1.2, ls="--", alpha=0.6)
    ax2.axhline(0.5, color="#94a3b8", lw=1.2, ls="--", alpha=0.6)
    ax2.set_xlabel("ΔE = E[layer46] − E[layer1]  (negative = decreased)", fontsize=11)
    ax2.set_ylabel("TM-score vs GT", fontsize=11)
    ax2.set_title(f"Energy drop vs fold quality\nAUROC={auc_delta:.3f}", fontsize=11)
    ax2.grid(True, lw=0.5)
    ax2.spines[["top","right"]].set_visible(False)

    out_png = ROOT / "runs/lj_energy_plot.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
