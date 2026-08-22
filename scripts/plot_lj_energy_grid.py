#!/usr/bin/env python3
"""Grid of per-protein LJ energy curves, ordered best→worst fold (TM score)."""
from __future__ import annotations
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
SIGMA = 6.8; EPS = 1.0; SEQ_SEP = 3; FIRST, LAST = 1, 46


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

def lj_energy(D):
    L = D.shape[0]
    ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
    d = D[ii, jj].clip(min=0.5)
    r = SIGMA / d
    return float((4 * EPS * (r**12 - r**6)).sum()) / L

def kabsch_tm(pred, gt):
    n = min(len(pred), len(gt))
    p, g = pred[:n].copy(), gt[:n].copy()
    p -= p.mean(0); gc = g - g.mean(0)
    H = p.T @ gc; U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1,1,np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    pa = p @ R.T + g.mean(0)
    d = np.sqrt(((pa - g)**2).sum(-1))
    d0 = max(0.5, 1.24*np.cbrt(max(n-15,1)) - 1.8)
    return float(np.mean(1/(1+(d/d0)**2)))


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

    print(f"Found {len(proteins)} proteins, computing energy curves...")

    records = []
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

        # short display name
        short = name.replace("benchmark_", "").replace("hard_synthetic_miss_", "✗") \
                    .replace("synthetic_design_", "★")
        records.append({"name": short, "tm": tm, "energies": np.array(energies)})

    # sort best → worst
    records.sort(key=lambda r: -r["tm"])
    N = len(records)

    # grid layout
    ncols = 8
    nrows = int(np.ceil(N / ncols))

    mpl.rcParams.update({
        "figure.facecolor": "#0f172a", "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155", "axes.labelcolor": "#94a3b8",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#1e293b", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.4, nrows * 2.2),
                             sharex=True)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.05,
                        wspace=0.30, hspace=0.55)
    fig.suptitle("LJ energy / residue across ESMFold layers  —  best → worst fold",
                 x=0.5, ha="center", fontsize=16, fontweight="bold", color="#f1f5f9")

    layers = np.arange(FIRST, LAST + 1)
    cmap   = plt.cm.RdYlGn

    axes_flat = axes.flatten()
    for i, rec in enumerate(records):
        ax  = axes_flat[i]
        tm  = rec["tm"]
        eng = rec["energies"]
        color = cmap(tm)

        ax.plot(layers, eng, color=color, lw=1.6)
        ax.fill_between(layers, eng, eng.min() - 0.5,
                        color=color, alpha=0.15)
        ax.set_title(f"{rec['name']}\nTM={tm:.2f}", fontsize=7.5,
                     color=color, pad=3)
        ax.set_ylim(eng.min() - 0.5, max(eng.max() + 0.3, 0.5))
        ax.tick_params(labelsize=6)
        ax.spines[["top","right"]].set_visible(False)
        # mark monotone fraction
        mono = float(np.mean(np.diff(eng) <= 0))
        ax.text(0.97, 0.08, f"↓{mono:.0%}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5,
                color="#94a3b8")

    # hide unused axes
    for j in range(N, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # shared x-label
    fig.text(0.5, 0.01, "ESMFold trunk layer", ha="center",
             fontsize=11, color="#94a3b8")

    out = ROOT / "runs/lj_energy_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
