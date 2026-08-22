#!/usr/bin/env python3
"""Steer ESMFold predictions by patching pair activations through a trained distogram probe.

Two modes:

  Single-pair:   steer selected residue pairs toward a target distance.
  Global:        steer every pair simultaneously toward a target MD conformation,
                 then measure RMSD(steered, target) vs RMSD(baseline, target).

Examples:
    # Single pair
    python scripts/steer_esmfold.py \\
        --probe-run-dir runs/probes_full_48 \\
        --activation-cache runs/steer_1ha8A00_cache.pt \\
        --inject-layer 20 --target-pairs "5,26" --target-dist 15.0

    # Global structure steering
    python scripts/steer_esmfold.py \\
        --probe-run-dir runs/probes_full_48 \\
        --activation-cache runs/steer_1ha8A00_cache.pt \\
        --inject-layer 20 --steer-global \\
        --mdcath-path data/mdcath/mdcath_dataset_1ha8A00.h5 \\
        --target-frame 0 \\
        --output-dir runs/steer_global_layer20
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import h5py
import numpy as np
import torch

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.geometry import pairwise_distances
from recurrent_folding.mdcath import parse_ca_atoms, read_scalar_text
from recurrent_folding.train_esmfold_distogram_probes import (
    ESMFOLD_DISTO_BINS,
    distogram_bin_centers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-run-dir", type=Path, required=True)
    p.add_argument("--activation-cache", type=Path, required=True)
    p.add_argument("--inject-layer", type=int, required=True,
                   help="Trunk block index (0–47) to inject steered z")
    p.add_argument("--probe-hidden-dim", type=int, default=256)
    p.add_argument("--steer-strength", type=float, default=10.0,
                   help="Peak height of the Gaussian logit bump")
    p.add_argument("--steer-sigma", type=float, default=2.0,
                   help="Width (σ) of the Gaussian steer in Å")
    p.add_argument("--output-dir", type=Path, default=Path("runs/steering"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Single-pair mode
    p.add_argument("--target-pairs", type=str, default=None,
                   help="Pairs to steer, e.g. '5,26' or '5,26+10,30'")
    p.add_argument("--target-dist", type=float, default=None,
                   help="Target Ca-Ca distance in Å (default: probe E[d] − 5)")

    # Global mode
    p.add_argument("--steer-global", action="store_true",
                   help="Steer all pairs simultaneously toward a target MD frame")
    p.add_argument("--mdcath-path", type=Path, default=None)
    p.add_argument("--domain", type=str, default=None)
    p.add_argument("--temperature", type=str, default="320")
    p.add_argument("--replica", type=str, default="0")
    p.add_argument("--target-frame", type=int, default=0,
                   help="MD frame index to use as steering target (default: 0)")
    return p.parse_args()


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_probe(probe_path: Path, pair_dim: int, hidden_dim: int, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=hidden_dim).to(device)
    probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
    probe.eval()
    for param in probe.parameters():
        param.requires_grad_(False)
    return probe


def load_target_ca_distances(
    mdcath_path: Path,
    domain: str | None,
    temperature: str,
    replica: str,
    frame: int,
) -> torch.Tensor:
    """Return (L, L) Ca-Ca distance matrix for a specific MD frame."""
    with h5py.File(mdcath_path, "r") as f:
        if domain is None:
            domain = next(iter(f.keys()))
        group = f[domain]
        ca_indices, _ = parse_ca_atoms(read_scalar_text(group["pdbProteinAtoms"]))
        coords_np = np.asarray(group[temperature][replica]["coords"][frame, ca_indices, :], dtype=np.float32)
    coords = torch.tensor(coords_np)
    return pairwise_distances(coords[None])[0]   # (L, L)


def steer_selected_pairs(
    z_layer: torch.Tensor,
    probe: PairDistogramBottleneck,
    pairs: list[tuple[int, int]],
    target_dist: float | None,
    strength: float,
    sigma: float,
    bin_centers: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict]]:
    L = z_layer.shape[0]
    z_steered = z_layer.clone()
    summaries = []

    for i, j in pairs:
        info: dict = {"pair": [i, j]}
        for ri, rj in [(i, j), (j, i)]:
            pair = z_layer[ri, rj][None].to(device)
            i_t = torch.tensor([ri], device=device)
            j_t = torch.tensor([rj], device=device)
            logits = probe.encode(pair, i_t, j_t, L)
            probs = torch.softmax(logits, dim=-1)
            expected_before = (probs * bin_centers.to(device)).sum(-1).item()
            td = target_dist if target_dist is not None else max(2.5, expected_before - 5.0)
            bump = strength * torch.exp(-0.5 * ((bin_centers.to(device) - td) / sigma) ** 2)
            steered_logits = logits + bump
            expected_after = (torch.softmax(steered_logits, -1) * bin_centers.to(device)).sum(-1).item()
            z_steered[ri, rj] = probe.decode(steered_logits, i_t, j_t, L)[0].cpu()
            if ri == i:
                info.update({"expected_dist_before": expected_before,
                              "target_dist": td, "expected_dist_after": expected_after})
        summaries.append(info)

    return z_steered, summaries


def steer_all_pairs(
    z_layer: torch.Tensor,        # (L, L, pair_dim) float32
    probe: PairDistogramBottleneck,
    target_dists: torch.Tensor,   # (L, L) target Ca-Ca distances
    strength: float,
    sigma: float,
    bin_centers: torch.Tensor,    # (1, bins)
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Steer every pair simultaneously toward target_dists using batch encoding."""
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_flat = ii.reshape(-1)
    jj_flat = jj.reshape(-1)

    pairs_flat = z_layer[ii_flat, jj_flat].to(device)                     # (L², pair_dim)
    logits = probe.encode(pairs_flat, ii_flat.to(device), jj_flat.to(device), L)  # (L², bins)

    probs_before = torch.softmax(logits, dim=-1)
    expected_before = (probs_before * bin_centers.to(device)).sum(-1)      # (L²,)

    target_flat = target_dists[ii_flat, jj_flat].to(device)               # (L²,)
    bump = strength * torch.exp(
        -0.5 * ((bin_centers.to(device) - target_flat[:, None]) / sigma) ** 2
    )
    steered_logits = logits + bump

    expected_after = (torch.softmax(steered_logits, -1) * bin_centers.to(device)).sum(-1)

    z_new_flat = probe.decode(steered_logits, ii_flat.to(device), jj_flat.to(device), L)
    z_steered = z_new_flat.reshape(L, L, -1).cpu()

    # Diagonal (self-pairs) — restore original to avoid NaN from zero-distance steering
    for k in range(L):
        z_steered[k, k] = z_layer[k, k]

    mean_repr_shift = (expected_after - expected_before).abs().mean().item()
    mean_target_shift = (target_flat - expected_before).abs().mean().item()
    print(f"  Avg representation shift: {mean_repr_shift:.2f}Å")
    print(f"  Avg gap to target before steering: {mean_target_shift:.2f}Å")

    return z_steered, {"mean_repr_shift": mean_repr_shift, "mean_target_shift": mean_target_shift}


class SteeringHook:
    def __init__(self, z_steered: torch.Tensor) -> None:
        self.z_steered = z_steered

    def __call__(self, _module, _args, output):
        s, z = output
        return (s, self.z_steered[None].to(device=z.device, dtype=z.dtype))


def run_esmfold(
    sequence: str,
    inject_block: int,
    device: torch.device,
    z_steered: torch.Tensor | None = None,
) -> np.ndarray:
    from transformers import AutoTokenizer, EsmForProteinFolding
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()

    ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    print("  Baseline forward pass...")
    with torch.no_grad():
        out_base = model(ids)
    ca_baseline = out_base["positions"][-1, 0, :, 1, :].cpu().numpy()

    ca_steered = None
    if z_steered is not None:
        print("  Steered forward pass...")
        hook = SteeringHook(z_steered)
        handle = model.trunk.blocks[inject_block].register_forward_hook(hook)
        with torch.no_grad():
            out_steered = model(ids)
        handle.remove()
        ca_steered = out_steered["positions"][-1, 0, :, 1, :].cpu().numpy()

    return ca_baseline, ca_steered


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    if not args.steer_global and args.target_pairs is None:
        raise ValueError("Provide --target-pairs for single-pair mode or --steer-global for global mode.")

    # Load activation cache
    print(f"Loading activation cache: {args.activation_cache}")
    cache = torch.load(args.activation_cache, map_location="cpu", weights_only=False)
    sequence: str = cache["sequence"]
    z_all = cache["z"].squeeze(1).contiguous()
    L, pair_dim = z_all.shape[1], z_all.shape[-1]
    print(f"  Sequence: {sequence}")
    print(f"  Length: {L}, pair_dim: {pair_dim}")

    probe_path = args.probe_run_dir / f"layer_{args.inject_layer:02d}" / "probe.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")
    print(f"Loading probe: {probe_path}")
    probe = load_probe(probe_path, pair_dim=pair_dim, hidden_dim=args.probe_hidden_dim, device=device)

    bin_centers = distogram_bin_centers(device, torch.float32)
    z_layer = z_all[args.inject_layer].float()

    summary: dict = {"inject_layer": args.inject_layer, "steer_strength": args.steer_strength}

    if args.steer_global:
        # ── Global mode ───────────────────────────────────────────────────────
        if args.mdcath_path is None:
            raise ValueError("--mdcath-path required for --steer-global")
        print(f"\nLoading target conformation (frame {args.target_frame}) from {args.mdcath_path}...")
        target_dists = load_target_ca_distances(
            args.mdcath_path, args.domain, args.temperature, args.replica, args.target_frame
        )
        if target_dists.shape[0] != L:
            raise ValueError(f"Target has {target_dists.shape[0]} residues but cache has {L}")

        baseline_to_target_rmsd = float(np.sqrt(np.mean(
            (target_dists.numpy() - target_dists.numpy()) ** 2  # placeholder — filled below
        )))

        print(f"\nSteering all {L}×{L} pairs at layer {args.inject_layer}...")
        z_steered, steer_info = steer_all_pairs(
            z_layer, probe, target_dists,
            strength=args.steer_strength,
            sigma=args.steer_sigma,
            bin_centers=bin_centers,
            device=device,
        )
        summary.update(steer_info)

        print("\nLoading ESMFold...")
        ca_baseline, ca_steered = run_esmfold(sequence, args.inject_layer, device, z_steered)

        # Compute distance matrices for RMSD comparison
        def dist_matrix(ca: np.ndarray) -> np.ndarray:
            d = ca[:, None, :] - ca[None, :, :]
            return np.sqrt((d ** 2).sum(-1))

        D_baseline = dist_matrix(ca_baseline)
        D_steered  = dist_matrix(ca_steered)
        D_target   = target_dists.numpy()

        mask = ~np.eye(L, dtype=bool)
        rmsd_bl_st  = float(np.sqrt(np.mean((D_baseline[mask] - D_steered[mask])  ** 2)))
        rmsd_bl_tgt = float(np.sqrt(np.mean((D_baseline[mask] - D_target[mask])   ** 2)))
        rmsd_st_tgt = float(np.sqrt(np.mean((D_steered[mask]  - D_target[mask])   ** 2)))
        frac_closed = 1.0 - rmsd_st_tgt / rmsd_bl_tgt if rmsd_bl_tgt > 0 else 0.0

        print("\n=== Global structure comparison ===")
        print(f"  RMSD(baseline, steered):  {rmsd_bl_st:.3f}Å   ← how much we moved the structure")
        print(f"  RMSD(baseline, target):   {rmsd_bl_tgt:.3f}Å   ← how far baseline was from target")
        print(f"  RMSD(steered,  target):   {rmsd_st_tgt:.3f}Å   ← how close steered got to target")
        print(f"  Fraction of gap closed:   {frac_closed:.3f}     ← 1.0 = perfect, 0.0 = no effect")

        summary.update({
            "target_frame": args.target_frame,
            "rmsd_baseline_steered": rmsd_bl_st,
            "rmsd_baseline_target": rmsd_bl_tgt,
            "rmsd_steered_target": rmsd_st_tgt,
            "fraction_gap_closed": frac_closed,
        })
        np.save(args.output_dir / "D_target.npy", D_target)

    else:
        # ── Single-pair mode ──────────────────────────────────────────────────
        pairs = [tuple(map(int, p.split(","))) for p in args.target_pairs.split("+")]
        print(f"\nSteering {len(pairs)} pair(s) at layer {args.inject_layer}...")
        z_steered, disto_summaries = steer_selected_pairs(
            z_layer, probe, pairs,
            target_dist=args.target_dist,
            strength=args.steer_strength,
            sigma=args.steer_sigma,
            bin_centers=bin_centers,
            device=device,
        )
        print("\n=== Distogram steering (pre-injection) ===")
        for info in disto_summaries:
            i, j = info["pair"]
            print(f"  ({i:3d},{j:3d})  probe E[d]: {info['expected_dist_before']:.2f}Å"
                  f" → {info['expected_dist_after']:.2f}Å  (target {info['target_dist']:.2f}Å)")

        print("\nLoading ESMFold...")
        ca_baseline, ca_steered = run_esmfold(sequence, args.inject_layer, device, z_steered)

        print("\n=== Structure comparison ===")
        results = []
        for info in disto_summaries:
            i, j = info["pair"]
            d_base  = float(np.linalg.norm(ca_baseline[i] - ca_baseline[j]))
            d_steer = float(np.linalg.norm(ca_steered[i]  - ca_steered[j]))
            delta = d_steer - d_base
            print(f"  Ca({i:3d})-Ca({j:3d}):  baseline {d_base:.2f}Å  "
                  f"steered {d_steer:.2f}Å  ({'+' if delta>=0 else ''}{delta:.2f}Å)"
                  f"  target {info['target_dist']:.2f}Å")
            results.append({**info, "baseline_ca_dist": d_base, "steered_ca_dist": d_steer})

        rmsd = float(np.sqrt(np.mean((ca_baseline - ca_steered) ** 2)))
        print(f"\n  Global Ca RMSD (baseline vs steered): {rmsd:.4f}Å")
        summary.update({"global_rmsd": rmsd, "pairs": results})

    np.save(args.output_dir / "ca_baseline.npy", ca_baseline)
    np.save(args.output_dir / "ca_steered.npy", ca_steered)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
