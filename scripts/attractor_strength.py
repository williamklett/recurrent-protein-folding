#!/usr/bin/env python3
"""Attractor-strength experiment.

For each protein:
  1) Sample K pairs (i,j) with |i-j| >= 5 and probe E[d] in [4, 18] Å.
  2) For each pair, steer the distogram at layer T by +5 Å (push longer).
  3) Run ESMFold from layer T+1.
  4) Measure how much the targeted pair's actual Cα-Cα distance moved
     relative to the baseline, and the global RMSD.

Predicted pattern:
  - Successes (deep basin): perturbations should be ABSORBED — small Δd, small global RMSD.
  - Failures (no/shallow basin): perturbations should PROPAGATE — large Δd, large RMSD.

Outputs:
  runs/attractor/<NAME>/results.csv             — per-pair measurements
  runs/attractor/<NAME>/summary.json            — per-protein aggregates
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from recurrent_folding.esmfold_probes import PairDistogramBottleneck


ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875
ESMFOLD_DISTO_BINS = 64


def bin_centers(device, dtype):
    edges = torch.linspace(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX, ESMFOLD_DISTO_BINS + 1,
                            device=device, dtype=dtype)
    return ((edges[:-1] + edges[1:]) * 0.5)


def steer_one_pair(
    z_layer: torch.Tensor,
    probe: PairDistogramBottleneck,
    i: int, j: int,
    delta_d: float,
    strength: float,
    sigma: float,
    bc: torch.Tensor,
    device,
) -> tuple[torch.Tensor, float, float]:
    """Return z with only pair (i,j)/(j,i) perturbed, plus the E[d] before/after."""
    z_new = z_layer.clone()
    L = z_layer.shape[0]
    expected_before = None
    expected_after = None
    for ri, rj in [(i, j), (j, i)]:
        pair = z_layer[ri, rj][None].to(device)
        i_t = torch.tensor([ri], device=device)
        j_t = torch.tensor([rj], device=device)
        logits = probe.encode(pair, i_t, j_t, L)
        probs = torch.softmax(logits, dim=-1)
        e_before = (probs * bc).sum(-1).item()
        target_d = float(np.clip(e_before + delta_d, ESMFOLD_DISTO_MIN + 0.5, ESMFOLD_DISTO_MAX - 0.5))
        bump = strength * torch.exp(-0.5 * ((bc - target_d) / sigma) ** 2)
        steered_logits = logits + bump
        e_after = (torch.softmax(steered_logits, -1) * bc).sum(-1).item()
        z_new[ri, rj] = probe.decode(steered_logits, i_t, j_t, L)[0].cpu()
        if ri == i:
            expected_before, expected_after = e_before, e_after
    return z_new, float(expected_before), float(expected_after)


class SteeringHook:
    def __init__(self, z_steered):
        self.z_steered = z_steered
    def __call__(self, _module, _args, output):
        s, z = output
        return (s, self.z_steered[None].to(device=z.device, dtype=z.dtype))


def sample_pairs(z_layer, probe, bc, device, rng, n_pairs: int) -> list[tuple[int, int]]:
    """Sample n pairs with |i-j|>=5 and probe E[d] in [4,18] Å."""
    L = z_layer.shape[0]
    chosen: list[tuple[int, int]] = []
    attempts = 0
    while len(chosen) < n_pairs and attempts < 500:
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        attempts += 1
        if abs(i - j) < 5:
            continue
        if (i, j) in chosen or (j, i) in chosen:
            continue
        pair = z_layer[i, j][None].to(device)
        i_t = torch.tensor([i], device=device)
        j_t = torch.tensor([j], device=device)
        with torch.no_grad():
            logits = probe.encode(pair, i_t, j_t, L)
        e = float((torch.softmax(logits, -1) * bc).sum(-1).item())
        if 4.0 <= e <= 18.0:
            chosen.append((i, j))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--inject-layer", type=int, default=20)
    ap.add_argument("--delta-d", type=float, default=5.0, help="Distogram push in Å (positive = longer)")
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--strength", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.run_dir.name.replace("benchmark_", "")
    rng = np.random.default_rng(args.seed)

    # ── Load cached z, sequence, baseline Ca ─────────────────────────────────
    # Memory-conscious load: extract only the injection layer's z then drop the
    # full (48, L, L, 128) tensor before ESMFold reloads its ~3 GB of weights.
    cache = torch.load(args.run_dir / "esmfold_activations.pt", map_location="cpu", weights_only=False)
    sequence = cache["sequence"]
    z_full = cache["z"].squeeze(1)         # (48, L, L, 128), original dtype (likely fp16)
    L = z_full.shape[1]
    pair_dim = z_full.shape[-1]
    z_layer_inject = z_full[args.inject_layer].float().contiguous()   # only the one we need
    del z_full, cache
    import gc; gc.collect()

    base_npz = args.run_dir / "analysis" / "esmfold_direct_eval" / "esmfold_prediction_ca.npz"
    if not base_npz.exists():
        print(f"  {name}: no baseline Ca cache, skipping")
        return
    ca_baseline = np.load(base_npz)["pred_ca"]  # (L, 3)
    if ca_baseline.shape[0] != L:
        print(f"  {name}: length mismatch baseline={ca_baseline.shape[0]} vs L={L}")
        return

    # ── Load probe at injection layer ────────────────────────────────────────
    inject_layer = args.inject_layer
    probe_path = args.run_dir / f"layer_{inject_layer:02d}" / "probe.pt"
    if not probe_path.exists():
        print(f"  {name}: no probe at layer {inject_layer}, skipping")
        return
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
    for p in probe.parameters(): p.requires_grad_(False)

    bc = bin_centers(device, torch.float32)
    # z_layer_inject already extracted above

    # ── Pick pairs ───────────────────────────────────────────────────────────
    pairs = sample_pairs(z_layer_inject, probe, bc, device, rng, args.n_pairs)
    print(f"  {name}: chose {len(pairs)} pairs at layer {inject_layer}: {pairs}")
    if not pairs:
        return

    # ── Load ESMFold once ────────────────────────────────────────────────────
    from transformers import AutoTokenizer, EsmForProteinFolding
    print(f"  {name}: loading ESMFold...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    model.esm = model.esm.half()
    ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    # ── For each pair: perturb, run, measure ─────────────────────────────────
    rows = []
    for k, (i, j) in enumerate(pairs):
        z_steered, e_before, e_after = steer_one_pair(
            z_layer_inject, probe, i, j,
            delta_d=args.delta_d,
            strength=args.strength,
            sigma=args.sigma,
            bc=bc,
            device=device,
        )
        hook = SteeringHook(z_steered)
        handle = model.trunk.blocks[inject_layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(ids)
        handle.remove()
        ca_steered = out["positions"][-1, 0, :, 1, :].cpu().numpy()

        d_pair_base = float(np.linalg.norm(ca_baseline[i] - ca_baseline[j]))
        d_pair_steer = float(np.linalg.norm(ca_steered[i] - ca_steered[j]))
        delta_pair = d_pair_steer - d_pair_base
        # Kabsch-aligned global RMSD
        def kabsch_rmsd(P, Q):
            Pc = P - P.mean(0); Qc = Q - Q.mean(0)
            H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1, 1, d]) @ U.T
            Pa = Pc @ R.T + Q.mean(0)
            return float(np.sqrt(((Pa - Q) ** 2).sum(-1).mean()))
        global_rmsd = kabsch_rmsd(ca_steered, ca_baseline)

        rows.append({
            "pair_i": i, "pair_j": j,
            "probe_e_before": e_before,
            "probe_e_after_steer": e_after,
            "applied_delta_e": e_after - e_before,
            "ca_dist_base": d_pair_base,
            "ca_dist_steered": d_pair_steer,
            "delta_pair_ca": delta_pair,
            "global_rmsd_vs_base": global_rmsd,
        })
        print(f"    pair ({i:3d},{j:3d})  probe E[d]: {e_before:.2f} -> {e_after:.2f} Å   "
              f"Ca dist: {d_pair_base:.2f} -> {d_pair_steer:.2f} ({delta_pair:+.2f})   "
              f"global RMSD: {global_rmsd:.3f}")

    # ── Write per-protein outputs ────────────────────────────────────────────
    pd = args.out_dir / name
    pd.mkdir(parents=True, exist_ok=True)
    with (pd / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

    # Responsiveness summary
    delta_e = np.array([r["applied_delta_e"] for r in rows])
    delta_pair = np.array([r["delta_pair_ca"] for r in rows])
    global_rmsd = np.array([r["global_rmsd_vs_base"] for r in rows])
    response_ratio = delta_pair / np.where(np.abs(delta_e) > 0.1, delta_e, np.nan)

    summary = {
        "name": name,
        "n_pairs": len(rows),
        "inject_layer": inject_layer,
        "mean_applied_delta_e": float(np.nanmean(delta_e)),
        "mean_abs_delta_pair": float(np.nanmean(np.abs(delta_pair))),
        "mean_delta_pair": float(np.nanmean(delta_pair)),
        "mean_response_ratio": float(np.nanmean(response_ratio)),
        "median_response_ratio": float(np.nanmedian(response_ratio)),
        "mean_global_rmsd": float(np.nanmean(global_rmsd)),
    }
    (pd / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  {name}: median response ratio = {summary['median_response_ratio']:+.3f}  "
          f"mean |Δpair| = {summary['mean_abs_delta_pair']:.3f} Å  "
          f"mean global RMSD = {summary['mean_global_rmsd']:.3f} Å")


if __name__ == "__main__":
    main()
