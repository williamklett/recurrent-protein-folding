#!/usr/bin/env python3
"""Quantify how much of the LJ(E[d]) drop is from real probability movement vs sharpening.

Three per-layer metrics:
  1. LJ(E[d])           — LJ at the mean (sensitive to sharpening)
  2. LJ(mode)           — LJ at the most-likely bin (robust to sharpening)
  3. E[LJ_capped(d)]    — full expectation with LJ capped at +5 (robust to repulsive blow-up)

Per-protein decomposition:
  ΔLJ_total     = LJ(E[d])_layer1  - LJ(E[d])_layer46
  ΔLJ_mode      = LJ(mode)_layer1  - LJ(mode)_layer46
  sharp_frac    = 1 - ΔLJ_mode / ΔLJ_total
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
SIGMA = 6.8; EPS = 1.0; SEQ_SEP = 3; FIRST, LAST = 1, 46
LJ_CAP = 5.0   # cap repulsive blow-up


def load_probe(layer, device):
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters(): p.requires_grad_(False)
    return pr


def lj(d: torch.Tensor) -> torch.Tensor:
    d = d.clamp(min=0.5)
    r = SIGMA / d
    return 4 * EPS * (r**12 - r**6)


def per_layer_metrics(z_layer, probe, centers, device):
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    sep = jj_f - ii_f
    keep = sep > SEQ_SEP
    ii_k, jj_k = ii_f[keep], jj_f[keep]

    pairs = z_layer[ii_k, jj_k].to(device)
    ii_d, jj_d = ii_k.to(device), jj_k.to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_d, jj_d, L)
        probs  = F.softmax(logits, dim=-1)

        # Flatten centers to 1D (it ships as [1,1,n_bins])
        c_flat = centers.reshape(-1)                           # (n_bins,)
        n_bins = probs.size(-1)
        c_used = c_flat[:n_bins].contiguous()                  # (n_bins,)
        lj_at_bins        = lj(c_used)                         # (n_bins,)
        lj_at_bins_capped = lj_at_bins.clamp(max=LJ_CAP)        # (n_bins,)

        ed       = (probs * c_used).sum(-1)                    # mean
        mode_bin = probs.argmax(-1).clamp(0, n_bins - 1)
        # gather LJ value at the mode bin for each pair
        lj_mode_per_pair = lj_at_bins.unsqueeze(0).expand(probs.size(0), n_bins).gather(
            1, mode_bin.unsqueeze(1)).squeeze(1)

        lj_mean   = lj(ed).sum().item() / L
        lj_mode_  = lj_mode_per_pair.sum().item() / L
        e_lj_cap  = (probs * lj_at_bins_capped).sum(-1).sum().item() / L

    return lj_mean, lj_mode_, e_lj_cap


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
    lj_bins         = lj(centers)
    lj_bins_capped  = lj_bins.clamp(max=LJ_CAP)

    proteins = []
    for act_p in sorted(RUNS_DIR.glob("*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p   = run_dir / "gt_ca_distances.npz"
        pred_p = next(run_dir.glob("**/esmfold_prediction_ca.npz"), None)
        if gt_p.exists() and pred_p:
            proteins.append((run_dir.name, act_p, gt_p, pred_p))

    n_layers = LAST - FIRST + 1
    all_mean   = np.zeros((len(proteins), n_layers))
    all_mode   = np.zeros((len(proteins), n_layers))
    all_capexp = np.zeros((len(proteins), n_layers))
    tms        = np.zeros(len(proteins))
    names      = []

    print(f"Found {len(proteins)} proteins")
    for pi, (name, act_p, gt_p, pred_p) in enumerate(proteins):
        print(f"[{pi+1}/{len(proteins)}] {name}", flush=True)
        act   = torch.load(act_p, map_location="cpu", weights_only=False)
        z_all = act["z"].squeeze(1).float()
        gt_ca   = np.load(gt_p)["ca_coords"].astype(float)
        pred_ca = np.load(pred_p)["pred_ca"].astype(float)
        tms[pi] = kabsch_tm(pred_ca, gt_ca)
        names.append(name)

        for t in range(FIRST, LAST + 1):
            probe = load_probe(t, device)
            m, mo, ce = per_layer_metrics(z_all[t], probe, centers, device)
            all_mean  [pi, t - FIRST] = m
            all_mode  [pi, t - FIRST] = mo
            all_capexp[pi, t - FIRST] = ce
            del probe
        torch.cuda.empty_cache()

    # Per-protein decomposition
    print(f"\n{'protein':28s} {'TM':>6} {'ΔLJ(E[d])':>11} {'ΔLJ(mode)':>11} {'ΔE[LJ_cap]':>11} {'sharp%':>8}")
    rows = []
    for pi in range(len(proteins)):
        d_mean = float(all_mean[pi, 0]  - all_mean[pi, -1])
        d_mode = float(all_mode[pi, 0]  - all_mode[pi, -1])
        d_cap  = float(all_capexp[pi,0] - all_capexp[pi,-1])
        if abs(d_mean) > 1e-6:
            sharp_frac = max(0.0, 1.0 - d_mode / d_mean)
        else:
            sharp_frac = float("nan")
        rows.append({"name": names[pi], "tm": float(tms[pi]),
                     "delta_lj_mean": round(d_mean, 4),
                     "delta_lj_mode": round(d_mode, 4),
                     "delta_lj_capexp": round(d_cap, 4),
                     "sharp_frac": round(sharp_frac, 4)})
        print(f"{names[pi]:28s} {tms[pi]:6.3f} {d_mean:11.3f} {d_mode:11.3f} "
              f"{d_cap:11.3f} {sharp_frac*100:7.1f}%")

    # Summary
    sharps  = np.array([r["sharp_frac"] for r in rows if np.isfinite(r["sharp_frac"])])
    good    = np.array([r["sharp_frac"] for r in rows if r["tm"] >= 0.7
                        and np.isfinite(r["sharp_frac"])])
    bad     = np.array([r["sharp_frac"] for r in rows if r["tm"] < 0.5
                        and np.isfinite(r["sharp_frac"])])
    print(f"\nMean sharpening contribution: {sharps.mean()*100:.1f}% "
          f"(median {np.median(sharps)*100:.1f}%)")
    print(f"  Good folds (TM≥0.7) sharp%: {good.mean()*100:.1f}% (n={len(good)})")
    print(f"  Bad  folds (TM<0.5) sharp%: {bad.mean()*100:.1f}%  (n={len(bad)})")

    out_csv = ROOT / "runs/lj_decomposition.csv"
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
    layers = np.arange(FIRST, LAST + 1)
    cmap   = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.83, bottom=0.11, wspace=0.28)
    fig.suptitle("How much of the LJ drop is real mass movement vs distribution sharpening?",
                 x=0.04, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color="#172033")
    fig.text(0.04, 0.89,
             f"Mean sharpening contribution: {sharps.mean()*100:.0f}%   "
             f"(good folds {good.mean()*100:.0f}%, bad folds {bad.mean()*100:.0f}%).  "
             f"N=57 proteins, coloured by TM.",
             ha="left", fontsize=10, color="#475569")

    for ax, title, arr in zip(
        axes[:3],
        ["LJ(E[d])  (sensitive to sharpening)",
         "LJ(mode)  (robust to sharpening)",
         f"E[LJ_capped(d)]  (cap={LJ_CAP})"],
        [all_mean, all_mode, all_capexp]):
        for pi in range(len(proteins)):
            ax.plot(layers, arr[pi], color=cmap(tms[pi]), lw=1.0, alpha=0.55)
        ax.plot(layers, arr.mean(0), color="#0f172a", lw=2.4, ls="--", label="mean")
        ax.set_xlabel("ESMFold trunk layer", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, lw=0.5)
        ax.legend(fontsize=9)

    # Histogram of sharp fractions
    ax4 = axes[3]
    ax4.hist(np.clip(sharps, 0, 1.5), bins=20, color="#3b82f6",
             edgecolor="#172033", alpha=0.85)
    ax4.axvline(sharps.mean(), color="#dc2626", lw=2,
                label=f"mean={sharps.mean()*100:.0f}%")
    ax4.set_xlabel("Fraction of ΔLJ explained by sharpening", fontsize=11)
    ax4.set_ylabel("# proteins", fontsize=11)
    ax4.set_title("Sharpening contribution per protein", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, lw=0.5, axis="y")

    out_png = ROOT / "runs/lj_decomposition.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
