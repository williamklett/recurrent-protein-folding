#!/usr/bin/env python3
"""Test whether LJ energy drop is a sharpness artifact.

Three quantities per layer per protein:
  1. LJ(E[d])       — LJ on the expected distance (original metric)
  2. E[LJ(d)]       — full expectation of LJ over the distogram (robust to sharpness)
  3. mean entropy   — average distogram entropy per pair (sharpness measure)

If LJ(E[d]) and E[LJ(d)] both decrease monotonically together, sharpness
is not the explanation. If entropy stays flat while LJ drops, even stronger.
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


def load_probe(layer, device):
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters(): p.requires_grad_(False)
    return pr


def lj_per_bin(centers: torch.Tensor) -> torch.Tensor:
    """LJ energy evaluated at each bin centre — shape (n_bins,)."""
    d = centers.clamp(min=0.5)
    r = SIGMA / d
    return 4.0 * EPS * (r**12 - r**6)


def per_layer_metrics(z_layer, probe, centers, lj_bins, device):
    """Returns (lj_of_mean, expected_lj, mean_entropy) for upper-triangle pairs."""
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)
    # Only upper triangle with |i-j| > SEQ_SEP
    sep = (jj_f - ii_f)
    keep = sep > SEQ_SEP
    ii_k, jj_k = ii_f[keep], jj_f[keep]

    pairs = z_layer[ii_k, jj_k].to(device)
    ii_d, jj_d = ii_k.to(device), jj_k.to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_d, jj_d, L)            # (N, n_bins)
        probs  = F.softmax(logits, dim=-1)                     # (N, n_bins)
        ed     = (probs * centers).sum(-1)                     # (N,)

        # 1. LJ evaluated at the mean predicted distance
        d = ed.clamp(min=0.5)
        r = SIGMA / d
        lj_of_mean = (4 * EPS * (r**12 - r**6))                # (N,)

        # 2. Full expectation of LJ over the distogram distribution
        expected_lj = (probs * lj_bins).sum(-1)                # (N,)

        # 3. Distogram entropy per pair (sharpness)
        eps = 1e-12
        entropy = -(probs * (probs + eps).log()).sum(-1)       # (N,)

    return (lj_of_mean.sum().item() / L,
            expected_lj.sum().item() / L,
            entropy.mean().item())


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
    centers = distogram_bin_centers(device, torch.float32)   # (n_bins,)
    lj_bins = lj_per_bin(centers)                            # (n_bins,)

    proteins = []
    for act_p in sorted(RUNS_DIR.glob("*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p   = run_dir / "gt_ca_distances.npz"
        pred_p = next(run_dir.glob("**/esmfold_prediction_ca.npz"), None)
        if gt_p.exists() and pred_p:
            proteins.append((run_dir.name, act_p, gt_p, pred_p))

    print(f"Found {len(proteins)} proteins")
    n_layers = LAST - FIRST + 1

    all_lj_mean   = np.zeros((len(proteins), n_layers))
    all_exp_lj    = np.zeros((len(proteins), n_layers))
    all_entropy   = np.zeros((len(proteins), n_layers))
    tms           = np.zeros(len(proteins))
    names         = []

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
            lj_m, exp_lj, ent = per_layer_metrics(z_all[t], probe, centers, lj_bins, device)
            all_lj_mean[pi, t - FIRST] = lj_m
            all_exp_lj [pi, t - FIRST] = exp_lj
            all_entropy[pi, t - FIRST] = ent
            del probe
        torch.cuda.empty_cache()

    # Save raw
    np.savez(ROOT / "runs/lj_sharpness_check.npz",
             names=np.array(names), tms=tms,
             lj_of_mean=all_lj_mean, expected_lj=all_exp_lj,
             entropy=all_entropy)
    print(f"saved npz")

    # Monotonicity stats
    def mono(arr): return float(np.mean(np.diff(arr) <= 0))
    print("\nPer-protein monotonicity (fraction of layer steps where metric decreases):")
    print("  protein                 TM    LJ(E[d])   E[LJ]    entropy")
    for pi in range(len(proteins)):
        print(f"  {names[pi]:24s} {tms[pi]:.3f}  "
              f"{mono(all_lj_mean[pi]):.3f}     {mono(all_exp_lj[pi]):.3f}    "
              f"{mono(all_entropy[pi]):.3f}")

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

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.10, wspace=0.25)
    fig.suptitle(
        "Sharpness check: does the LJ energy drop survive when computed as a full expectation?",
        x=0.05, y=0.96, ha="left", fontsize=15,
        fontweight="bold", color="#172033")
    fig.text(0.05, 0.89,
             "If LJ(E[d]) and E[LJ] decrease together while entropy stays roughly flat, "
             "sharpness is not the artifact.  N=57 proteins, coloured by TM.",
             ha="left", fontsize=10, color="#475569")

    titles  = ["LJ(E[d])  (original)", "E[LJ(d)]  (sharpness-robust)", "Distogram entropy"]
    arrays  = [all_lj_mean, all_exp_lj, all_entropy]
    for ax, title, arr in zip(axes, titles, arrays):
        for pi in range(len(proteins)):
            ax.plot(layers, arr[pi], color=cmap(tms[pi]), lw=1.0, alpha=0.6)
        ax.plot(layers, arr.mean(0), color="#0f172a", lw=2.6, ls="--",
                label="mean")
        ax.set_xlabel("ESMFold trunk layer", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, lw=0.5)
        ax.legend(fontsize=9)

    out = ROOT / "runs/lj_sharpness_check.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
