#!/usr/bin/env python3
"""Triangle inequality violation rate vs layer.

For each layer, run the cw30 probe on cached z representations,
get E[d] per pair, sample random triples (i,j,k) and check what
fraction violate d(i,k) > d(i,j) + d(j,k).

If probes read a coherent 3D frame, violations should be rare.
If probes were independent per-pair re-encodings, violations would be frequent.
No cross-pair term in the loss — so any Euclidean structure is emergent.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT   = Path(__file__).resolve().parents[1]
PROBES = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS   = ROOT / "runs"

N_TRIPLES = 5000   # random triples to sample per protein per layer
N_LAYERS  = 47


def load_probe(layer: int, device: torch.device) -> PairDistogramBottleneck:
    path = PROBES / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(
        pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256
    ).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def expected_dist_matrix(z_layer: torch.Tensor, probe: PairDistogramBottleneck,
                         centers: torch.Tensor, device: torch.device) -> np.ndarray:
    """Returns (L, L) matrix of E[d] from probe predictions."""
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f = ii.reshape(-1).to(device)
    jj_f = jj.reshape(-1).to(device)
    pairs = z_layer[ii_f, jj_f].to(device)
    with torch.no_grad():
        logits = probe.encode(pairs, ii_f, jj_f, L)
        probs  = F.softmax(logits, dim=-1)
        ed     = (probs * centers).sum(-1)
    return ed.reshape(L, L).cpu().numpy()


def violation_rate(D: np.ndarray, n_triples: int, rng: np.random.Generator) -> float:
    """Fraction of random triples violating triangle inequality."""
    L = D.shape[0]
    if L < 3:
        return float("nan")
    idx = rng.integers(0, L, size=(n_triples, 3))
    # ensure distinct
    mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    idx  = idx[mask]
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    # check all three orientations
    v1 = D[i, k] > D[i, j] + D[j, k]
    v2 = D[i, j] > D[i, k] + D[k, j]
    v3 = D[j, k] > D[j, i] + D[i, k]
    return float((v1 | v2 | v3).mean())


def find_proteins() -> list[Path]:
    """Find run dirs with both esmfold_activations.pt and gt_ca_distances.npz."""
    found = []
    for act_p in sorted(RUNS.glob("benchmark_*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p = run_dir / "gt_ca_distances.npz"
        if gt_p.exists():
            found.append(run_dir)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=6)
    ap.add_argument("--n-triples",  type=int, default=N_TRIPLES)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/triangle_inequality_by_layer.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    centers = distogram_bin_centers(device, torch.float32)
    rng     = np.random.default_rng(42)

    proteins = find_proteins()[:args.n_proteins]
    print(f"Using {len(proteins)} proteins:")
    for p in proteins:
        print(f"  {p.name}")

    # violations[layer] = list of rates across proteins
    all_rates = np.zeros((N_LAYERS, len(proteins)))

    for pi, run_dir in enumerate(proteins):
        name = run_dir.name
        print(f"\n[{pi+1}/{len(proteins)}] {name}")
        act  = torch.load(run_dir / "esmfold_activations.pt",
                          map_location="cpu", weights_only=False)
        z_all = act["z"].squeeze(1).float()   # (48, L, L, 128)

        for t in range(N_LAYERS):
            probe = load_probe(t, device)
            D = expected_dist_matrix(z_all[t], probe, centers, device)
            all_rates[t, pi] = violation_rate(D, args.n_triples, rng)
            del probe
            if t % 10 == 0:
                print(f"  layer {t:2d}  violation={all_rates[t, pi]:.4f}")

        torch.cuda.empty_cache()

    mean_rates = all_rates.mean(axis=1)
    std_rates  = all_rates.std(axis=1)

    print("\n=== Triangle inequality violation rate by layer ===")
    for t in range(N_LAYERS):
        print(f"  layer {t:2d}: {mean_rates[t]:.4f} ± {std_rates[t]:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })
    layers = np.arange(N_LAYERS)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)
    fig.suptitle(
        "Triangle inequality violation rate — probe-predicted distances vs layer",
        x=0.09, y=0.95, ha="left", fontsize=15, fontweight="bold", color="#172033")
    fig.text(
        0.09, 0.875,
        f"E[d] per pair from cw30 probe.  "
        f"Probe loss is per-pair KL + recon — no cross-pair consistency term.  "
        f"Low violation → probes read a shared Euclidean frame from z (faithfulness).  "
        f"n={len(proteins)} proteins, {args.n_triples} random triples/layer.",
        ha="left", fontsize=10, color="#475569")

    ax.plot(layers, mean_rates, color="#2563eb", lw=2.4, zorder=3)
    ax.fill_between(layers,
                    mean_rates - std_rates,
                    mean_rates + std_rates,
                    color="#2563eb", alpha=0.15)
    for pi in range(len(proteins)):
        ax.plot(layers, all_rates[:, pi], color="#94a3b8", lw=0.8, alpha=0.5)

    ax.set_xlabel("ESMFold trunk layer", fontsize=12)
    ax.set_ylabel("Triangle inequality violation rate", fontsize=12)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
