#!/usr/bin/env python3
"""Cosine alignment between ESMFold's per-layer update Δz_t and −∇_z L(z_t) for
several candidate "objective" losses L. The question: is the trunk implementing
gradient descent on some objective the probe can read?

Per benchmark protein, for each layer t in 1..46:
  Δz_t      = z_{t+1} − z_t      (actual trunk update at layer t)
  ∇L_ent    = ∇_z entropy of probe_t's distogram at z_t      (self-confidence)
  ∇L_gt     = ∇_z CE against ground-truth distance bins      (truth-aligned)
  ∇L_final  = ∇_z CE against layer-47 probe's distogram      (self-prediction)

Outputs per layer for each loss:
  cos(Δz_t, −∇L) and a random-direction baseline.

If GD is the right story, cos(Δz_t, −∇L_gt) should be > 0 for successes
and ≈ 0 for failures (failures can't be doing GD on truth they don't have).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck


ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875
ESMFOLD_DISTO_BINS = 64


def soft_bin_targets(D: torch.Tensor) -> torch.Tensor:
    """Map a distance matrix to bin indices (shape (L,L) -> long (L*L,))."""
    Dc = D.clamp(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX - 1e-3)
    binw = (ESMFOLD_DISTO_MAX - ESMFOLD_DISTO_MIN) / ESMFOLD_DISTO_BINS
    idx = ((Dc - ESMFOLD_DISTO_MIN) / binw).long()
    return idx.clamp(0, ESMFOLD_DISTO_BINS - 1).flatten()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))


def grad_of_loss(z_t: torch.Tensor, probe, i_t, j_t, L, loss_fn) -> torch.Tensor:
    """Return ∇_z loss_fn(logits(z_t)) flattened to a vector."""
    z = z_t.detach().clone().requires_grad_(True)
    logits = probe.encode(z.reshape(-1, z.shape[-1]), i_t, j_t, L)
    loss = loss_fn(logits)
    grad = torch.autograd.grad(loss, z)[0]
    return grad.flatten()


def compute_for_protein(run_dir: Path, device: str = "cuda") -> list[dict]:
    cache_path = run_dir / "esmfold_activations.pt"
    gt_path = run_dir / "gt_ca_distances.npz"
    if not cache_path.exists() or not gt_path.exists():
        return []

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    z_all = cache["z"].squeeze(1).float()  # (48, L, L, 128)
    n_layers = z_all.shape[0]   # should be 48
    L = z_all.shape[1]
    pair_dim = z_all.shape[-1]

    gt = np.load(gt_path)
    # Length-align GT distances to model length (already done in pipeline but be safe).
    D_gt_np = gt["distances"]
    n = min(D_gt_np.shape[0], L)
    D_gt = torch.from_numpy(D_gt_np[:n, :n]).float().to(device)
    if n != L:
        # The probe operates on the full LxL grid; pad with zeros outside is wrong.
        # We bail rather than introduce a silent bug.
        print(f"  [{run_dir.name}] length mismatch L={L} vs GT={n}, skipping")
        return []

    gt_targets = soft_bin_targets(D_gt).to(device)

    i_idx, j_idx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    i_t = i_idx.flatten().to(device).long()
    j_t = j_idx.flatten().to(device).long()

    # Pre-load the layer-47 probe to compute the "self-prediction" target distogram.
    probe47_path = run_dir / "layer_47" / "probe.pt"
    if not probe47_path.exists():
        return []
    probe47 = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe47.load_state_dict(torch.load(probe47_path, map_location=device, weights_only=True))
    for p in probe47.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        z47 = z_all[-1].to(device)
        logits47 = probe47.encode(z47.reshape(-1, pair_dim), i_t, j_t, L)
        p47 = F.softmax(logits47, dim=-1).detach()

    rows = []
    rng = torch.Generator(device=device).manual_seed(0)

    # z_all is indexed 0..47, layers numbered 1..47 in the pipeline use z_index = layer-1.
    # We compare Δz at layer t = z[t] - z[t-1] for t in 1..47.
    # For probe-loss gradients we use the probe trained at layer t (i.e. on z[t-1]).
    for layer in range(1, n_layers):
        probe_path = run_dir / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists():
            continue

        probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
        probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
        for p in probe.parameters():
            p.requires_grad_(False)

        # z_t is the state going INTO layer `layer`, i.e. z_all[layer-1].
        z_cur = z_all[layer - 1].to(device)
        z_next = z_all[layer].to(device) if layer < n_layers else None
        if z_next is None:
            continue
        delta_z = (z_next - z_cur).flatten()
        delta_norm = float(delta_z.norm())
        if delta_norm < 1e-9:
            continue

        # Loss 1: entropy of probe distogram
        def loss_entropy(logits):
            p = F.softmax(logits, dim=-1)
            return -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1).mean()

        # Loss 2: CE against ground-truth distance bins
        def loss_gt(logits):
            return F.cross_entropy(logits, gt_targets, reduction="mean")

        # Loss 3: CE against layer-47 probe's soft distogram (self-prediction)
        def loss_final(logits):
            logp = F.log_softmax(logits, dim=-1)
            return -(p47 * logp).sum(dim=-1).mean()

        g_ent  = grad_of_loss(z_cur, probe, i_t, j_t, L, loss_entropy)
        g_gt   = grad_of_loss(z_cur, probe, i_t, j_t, L, loss_gt)
        g_fin  = grad_of_loss(z_cur, probe, i_t, j_t, L, loss_final)

        # Random direction baseline (Gaussian with same scale as delta)
        rand_dir = torch.randn(delta_z.shape, device=device, generator=rng)

        rows.append({
            "layer":             layer,
            "cos_neg_grad_ent":  cosine(delta_z, -g_ent),
            "cos_neg_grad_gt":   cosine(delta_z, -g_gt),
            "cos_neg_grad_fin":  cosine(delta_z, -g_fin),
            "cos_random":        cosine(delta_z, rand_dir),
            "norm_delta_z":      delta_norm,
            "norm_grad_ent":     float(g_ent.norm()),
            "norm_grad_gt":      float(g_gt.norm()),
            "norm_grad_fin":     float(g_fin.norm()),
        })

        # Free probe memory before next layer
        del probe
        torch.cuda.empty_cache()

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--device",  default="cuda")
    args = ap.parse_args()

    rows = compute_for_protein(args.run_dir, args.device)
    if not rows:
        print(f"  no rows for {args.run_dir}")
        return
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {args.out_csv}  ({len(rows)} layers)")


if __name__ == "__main__":
    main()
