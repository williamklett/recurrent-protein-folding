#!/usr/bin/env python3
"""Zero-shot cross-protein generalization test for distogram probes.

Runs ESMFold on a new protein, then evaluates probes trained on a different protein
(1ha8A00) without any retraining. Reports per-layer:
  - R²   : reconstruction quality (z → distogram → z') on the new protein
  - corr : Pearson r between E[d]_probe@T and final ESMFold Ca-Ca distances

If probes learned universal geometry, both should stay high across layers.
If they learned protein-specific codes, both will drop on the new protein.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck, ESMFoldTrunkCapture
from recurrent_folding.train_esmfold_distogram_probes import (
    ESMFOLD_DISTO_BINS,
    distogram_bin_centers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-run-dir", type=Path, required=True,
                   help="Dir with per-layer probe checkpoints trained on source protein")
    p.add_argument("--sequence", type=str, required=True,
                   help="Amino acid sequence of the new (evaluation) protein")
    p.add_argument("--probe-hidden-dim", type=int, default=256)
    p.add_argument("--pairs-per-layer", type=int, default=8192,
                   help="Number of randomly sampled pairs for R² estimation per layer")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to save results JSON")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def run_esmfold_and_cache(sequence: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    """Returns z_all (48, L, L, 128) and ca_coords (L, 3)."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    print("Loading ESMFold...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()

    ids = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    capture = ESMFoldTrunkCapture(model)

    t0 = time.time()
    with capture.capturing(), torch.no_grad():
        out = model(ids)
    print(f"  Forward pass: {time.time()-t0:.1f}s")

    z_all = capture.stacked_z().squeeze(1).float()   # (48, L, L, 128)
    ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()  # (L, 3)
    return z_all, ca


def load_probe(path: Path, pair_dim: int, hidden_dim: int, device: torch.device) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=hidden_dim).to(device)
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def eval_layer(
    z_layer: torch.Tensor,        # (L, L, pair_dim)
    ca: np.ndarray,               # (L, 3)
    probe: PairDistogramBottleneck,
    bin_centers: torch.Tensor,    # (1, bins)
    n_pairs: int,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    L = z_layer.shape[0]

    # Sample off-diagonal pairs
    all_i, all_j = np.triu_indices(L, k=1)
    n = min(n_pairs, len(all_i))
    idx = rng.choice(len(all_i), size=n, replace=False)
    si, sj = all_i[idx], all_j[idx]
    # add both directions
    ii = np.concatenate([si, sj])
    jj = np.concatenate([sj, si])

    i_t = torch.from_numpy(ii).long().to(device)
    j_t = torch.from_numpy(jj).long().to(device)
    pairs = z_layer[ii, jj].to(device)              # (2n, pair_dim)

    with torch.no_grad():
        logits, reconstructed = probe(pairs, i_t, j_t, L)

    # R²
    recon_mse = F.mse_loss(reconstructed, pairs).item()
    recon_var  = pairs.var().item()
    r2 = 1.0 - recon_mse / max(recon_var, 1e-8)

    # E[d] from probe vs actual Ca-Ca
    ed = (torch.softmax(logits, dim=-1) * bin_centers.to(device)).sum(-1).cpu().numpy()  # (2n,)

    # Actual distances
    diff = ca[ii] - ca[jj]
    actual_dist = np.sqrt((diff ** 2).sum(-1))   # (2n,)

    # Pearson correlation
    if actual_dist.std() > 1e-6 and ed.std() > 1e-6:
        corr = float(np.corrcoef(ed, actual_dist)[0, 1])
    else:
        corr = 0.0

    return {"r2": r2, "corr": corr, "recon_mse": recon_mse, "recon_var": recon_var}


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    rng = np.random.default_rng(42)

    print(f"Sequence: {args.sequence}")
    print(f"Length:   {len(args.sequence)}")

    z_all, ca = run_esmfold_and_cache(args.sequence, device)
    L, pair_dim = z_all.shape[1], z_all.shape[-1]
    print(f"z_all shape: {z_all.shape}")

    bin_centers = distogram_bin_centers(device, torch.float32)

    print(f"\n{'layer':>6}  {'R²':>6}  {'corr':>6}  {'recon_mse':>10}  {'recon_var':>10}  probe")
    print("-" * 60)

    results = []
    for layer in range(47, -1, -1):
        probe_path = args.probe_run_dir / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists():
            print(f"{layer:6d}  MISSING")
            continue

        probe = load_probe(probe_path, pair_dim=pair_dim, hidden_dim=args.probe_hidden_dim, device=device)
        z_layer = z_all[layer]

        m = eval_layer(z_layer, ca, probe, bin_centers, args.pairs_per_layer, device, rng)
        status = "ok" if probe_path.exists() else "missing"
        print(f"{layer:6d}  {m['r2']:6.3f}  {m['corr']:6.3f}  {m['recon_mse']:10.3f}  {m['recon_var']:10.3f}")
        results.append({"layer": layer, **m})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"sequence": args.sequence, "results": results}, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
