#!/usr/bin/env python3
"""
Estimate the implicit gradient norm ||∂D/∂z|| along the trunk trajectory.

Intuition (first-principles):
  ESMFold's trunk performs iterative inference. Near a fixed point of the
  implicit objective, the derivative of the readout (distogram D) with respect
  to the underlying state (z) goes to zero. The state itself can still drift
  in null directions (gauge freedom in z that doesn't affect D), but the
  *readout* stops moving.

Estimator (finite-difference, last-8-block window):
  ΔD_t / Δz_t   averaged over the last 8 blocks
  where
    Δz_t = mean over pairs of ||z_{t+1}[i,j] - z_t[i,j]||_2           (z-units)
    ΔD_t = mean over pairs of |E[d_{t+1}](i,j) - E[d_t](i,j)|         (Å per pair)

Reading the ratio:
  • Small (≈ 0) → readout has stopped responding to z motion → converged
  • Large       → every z step drives a proportional D step      → still descending

This is theory-clean: it's a discrete approximation of the Jacobian norm of
the implicit readout function evaluated near the trunk's terminal state.

Local availability: requires both cached z (ESMFold trunk activations) and
trained probes. Only 1ha8A00 has both locally — used as proof-of-concept.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from recurrent_folding.esmfold_probes import PairDistogramBottleneck

# Inline these constants to avoid pulling in h5py via the training module
ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875
ESMFOLD_DISTO_BINS = 64


def distogram_bin_centers(device, dtype):
    edges = torch.linspace(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX,
                           ESMFOLD_DISTO_BINS + 1, device=device, dtype=dtype)
    return ((edges[:-1] + edges[1:]) * 0.5).view(1, -1)

ROOT = Path(__file__).parent.parent


def load_z_cache(path: Path) -> tuple[torch.Tensor, str]:
    """Returns z (48, L, L, 128), float32, and the sequence."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    z = data["z"].squeeze(1).float()   # (48, L, L, 128)
    return z, data["sequence"]


def load_probe(path: Path, pair_dim: int, hidden_dim: int = 256) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=hidden_dim)
    probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def expected_distances(probe: PairDistogramBottleneck,
                       z_layer: torch.Tensor,
                       bin_centers: torch.Tensor) -> torch.Tensor:
    """Run probe and return (L, L) matrix of expected distances E[d]."""
    L = z_layer.shape[0]
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    i_t = torch.from_numpy(ii.flatten()).long()
    j_t = torch.from_numpy(jj.flatten()).long()
    pairs = z_layer.reshape(-1, z_layer.shape[-1])    # (L², D)
    with torch.no_grad():
        logits = probe.encode(pairs, i_t, j_t, L)     # (L², bins)
    probs = torch.softmax(logits, dim=-1)
    ed = (probs * bin_centers).sum(-1)                # (L²,)
    return ed.reshape(L, L)


def main() -> None:
    z_cache = ROOT / "runs/remote/esmfold_probe_real_smoke/esmfold_activations.pt"
    probe_dir = ROOT / "runs/remote/esmfold_probe_20layers_real"

    if not z_cache.exists() or not probe_dir.exists():
        print(f"Missing inputs:")
        print(f"  z cache:   {z_cache.exists()}  ({z_cache})")
        print(f"  probe dir: {probe_dir.exists()}  ({probe_dir})")
        sys.exit(1)

    z, seq = load_z_cache(z_cache)
    L = z.shape[1]; pair_dim = z.shape[-1]
    print(f"Sequence : {seq}")
    print(f"z shape  : {tuple(z.shape)}  (48 blocks × {L}×{L}×{pair_dim})")

    bin_centers = distogram_bin_centers(torch.device("cpu"), torch.float32)
    bin_centers = bin_centers.squeeze(0)   # (bins,)

    # Compute distogram E[d] at each layer using its own probe
    print("\nComputing distograms at each layer ...")
    D_per_layer = []
    for layer in range(48):
        probe_path = probe_dir / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists():
            D_per_layer.append(None)
            continue
        probe = load_probe(probe_path, pair_dim)
        ed = expected_distances(probe, z[layer], bin_centers)  # (L, L)
        D_per_layer.append(ed)

    # Find consecutive layers where both have probes
    layers_available = [t for t, d in enumerate(D_per_layer) if d is not None]
    print(f"Layers with probes: {layers_available[0]}..{layers_available[-1]} "
          f"(N={len(layers_available)})")

    # Per-pair upper triangle (excluding diagonal)
    iu = np.triu_indices(L, k=1)
    iu_t = (torch.from_numpy(iu[0]), torch.from_numpy(iu[1]))

    # Step sizes
    dz_list:   list[float] = []
    dD_list:   list[float] = []
    ratio_list: list[float] = []
    layer_idx: list[int]   = []

    for t in layers_available[:-1]:
        t1 = t + 1
        if t1 not in layers_available:
            continue
        # Δz per pair: L2 norm of (z[t+1][i,j] - z[t][i,j]) in 128-dim
        dz_pairs = (z[t1] - z[t]).pow(2).sum(-1).sqrt()     # (L, L)
        dz_mean  = float(dz_pairs[iu_t].mean())
        # ΔD per pair: |E[d_{t+1}] - E[d_t]| in Å
        dD_pairs = (D_per_layer[t1] - D_per_layer[t]).abs() # (L, L)
        dD_mean  = float(dD_pairs[iu_t].mean())

        dz_list.append(dz_mean)
        dD_list.append(dD_mean)
        ratio_list.append(dD_mean / max(dz_mean, 1e-12))
        layer_idx.append(t)

    dz_arr = np.array(dz_list)
    dD_arr = np.array(dD_list)
    ratio  = np.array(ratio_list)

    # ── Print per-layer table ──────────────────────────────────────────────
    print(f"\n{'t→t+1':>8}  {'Δz':>9}  {'ΔD (Å)':>9}  {'ΔD/Δz':>10}")
    print("─" * 44)
    for i, t in enumerate(layer_idx):
        marker = "  ← last 8" if i >= len(layer_idx) - 8 else ""
        print(f"{t:3d}→{t+1:<4d}  {dz_arr[i]:9.4f}  {dD_arr[i]:9.4f}  {ratio[i]:10.4f}{marker}")

    last8 = ratio[-8:]
    last16 = ratio[-16:]
    first8 = ratio[:8]

    print(f"\nGradient-estimate summary (1ha8A00, a folded protein):")
    print(f"  first 8 blocks:  ΔD/Δz mean = {first8.mean():.4f}  (std {first8.std():.4f})")
    print(f"  last 16 blocks:  ΔD/Δz mean = {last16.mean():.4f}  (std {last16.std():.4f})")
    print(f"  last  8 blocks:  ΔD/Δz mean = {last8.mean():.4f}   (std {last8.std():.4f})")
    print(f"  ratio last/first ≈ {last8.mean()/max(first8.mean(),1e-9):.3f}")
    print(f"\n  (small last-8 ratio + small last/first ratio ⇒ trunk approached a "
          f"fixed point in the readout subspace)")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = ROOT / "runs" / "breathing_analysis"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    ax.plot(layer_idx, dz_arr, marker="o", color="#1976D2", lw=1.5, ms=4)
    ax.set_ylabel("Δz per pair (L2 in z-units)")
    ax.set_title("1ha8A00 — z step size between consecutive trunk blocks")
    ax.grid(alpha=0.3)
    ax.axvspan(layer_idx[-8], layer_idx[-1], alpha=0.15, color="orange", label="last 8")
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(layer_idx, dD_arr, marker="o", color="#388E3C", lw=1.5, ms=4)
    ax.set_ylabel("ΔE[d] per pair (Å)")
    ax.set_title("Distogram step size (probe readout)")
    ax.grid(alpha=0.3)
    ax.axvspan(layer_idx[-8], layer_idx[-1], alpha=0.15, color="orange")

    ax = axes[2]
    ax.plot(layer_idx, ratio, marker="o", color="#D32F2F", lw=1.5, ms=4)
    ax.set_ylabel("ΔD / Δz  (Å per z-unit)")
    ax.set_xlabel("ESMFold trunk block t  (Δ between t and t+1)")
    ax.set_title("Estimated implicit gradient norm — should drop toward 0 at convergence")
    ax.grid(alpha=0.3)
    ax.axvspan(layer_idx[-8], layer_idx[-1], alpha=0.15, color="orange")

    plt.tight_layout()
    p = out / "fig6_gradient_estimate_1ha8A00.png"
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
