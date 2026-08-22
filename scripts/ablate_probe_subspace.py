#!/usr/bin/env python3
"""Probe subspace ablation: how causally important is each layer's distogram subspace?

For every trunk block T that has a trained probe:

  1. Run ESMFold normally; capture z_T (pair activations at block T).
  2. Project through the probe bottleneck:
         z_hat = probe.decode(probe.encode(z_T))
  3. Re-run ESMFold with a hook at block T that injects
         z_residual = z_T  -  z_hat
     (i.e. everything EXCEPT what the probe captures).
  4. Measure Ca-RMSD(baseline, ablated).

Interpretation
--------------
  High RMSD at layer T  ->  the probe's subspace IS causally important there;
                             removing it shifts the final structure.
  Low  RMSD at layer T  ->  the model ignores the probe's subspace at that layer;
                             structural information lives elsewhere in z.

Combined with probe_r2 (how well the probe explains z_T):
  High r2 + High RMSD  ->  probe is a faithful causal handle.
  High r2 + Low  RMSD  ->  probe reads out geometry but doesn't drive it.
  Low  r2 + any  RMSD  ->  probe barely captures z_T; ablation uninformative.
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
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck, ESMFoldTrunkCapture
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--probe-run-dir", type=Path, required=True,
                   help="Directory containing per-layer probe checkpoints "
                        "(layer_00/probe.pt … layer_47/probe.pt)")
    p.add_argument("--sequence", type=str, required=True,
                   help="Amino-acid sequence of the protein to evaluate")
    p.add_argument("--probe-hidden-dim", type=int, default=256)
    p.add_argument("--output-dir", type=Path, default=Path("runs/ablate_probe"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--inject-zhat", action="store_true",
                   help="Inject z_hat (probe reconstruction) instead of z_residual "
                        "(z - z_hat). Use as the dual control experiment.")
    return p.parse_args()


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ---------------------------------------------------------------------------
# ESMFold helpers
# ---------------------------------------------------------------------------

def load_esmfold(device: torch.device):
    from transformers import AutoTokenizer, EsmForProteinFolding

    print("Loading ESMFold …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    return tokenizer, model


def run_baseline(
    tokenizer, model, sequence: str, device: torch.device
) -> tuple[np.ndarray, torch.Tensor]:
    """Normal forward pass.

    Returns:
        ca_baseline : (L, 3) Ca coordinates
        z_all       : (48, L, L, 128) pair activations, float32
    """
    ids = tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    capture = ESMFoldTrunkCapture(model)
    t0 = time.time()
    with capture.capturing(), torch.no_grad():
        out = model(ids)
    print(f"  Baseline forward: {time.time() - t0:.1f}s")

    ca = out["positions"][-1, 0, :, 1, :].cpu().numpy()   # (L, 3)
    z_all = capture.stacked_z().squeeze(1).float()         # (48, L, L, 128)
    return ca, z_all


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def load_probe(
    path: Path, pair_dim: int, hidden_dim: int, device: torch.device
) -> PairDistogramBottleneck:
    probe = PairDistogramBottleneck(
        pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=hidden_dim
    ).to(device)
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    probe.eval()
    for param in probe.parameters():
        param.requires_grad_(False)
    return probe


def compute_probe_split(
    z_layer: torch.Tensor,          # (L, L, pair_dim) cpu
    probe: PairDistogramBottleneck,
    device: torch.device,
    inject_zhat: bool = False,
) -> tuple[torch.Tensor, float]:
    """Split z_layer into probe-explained (z_hat) and residual (z - z_hat).

    Args:
        inject_zhat: if True return z_hat; if False return z - z_hat.

    Returns:
        z_inject : (L, L, pair_dim) on device
        r2       : float — probe reconstruction R²
    """
    L = z_layer.shape[0]

    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    i_t = torch.from_numpy(ii.flatten()).long().to(device)
    j_t = torch.from_numpy(jj.flatten()).long().to(device)
    pairs = z_layer.reshape(-1, z_layer.shape[-1]).to(device)   # (L², D)

    with torch.no_grad():
        _logits, reconstructed = probe(pairs, i_t, j_t, L)     # (L², D)

    z_hat = reconstructed.reshape(L, L, -1)                     # (L, L, D)
    z_inject = z_hat if inject_zhat else (z_layer.to(device) - z_hat)

    mse = F.mse_loss(reconstructed, pairs).item()
    var = pairs.var().item()
    r2  = 1.0 - mse / max(var, 1e-8)

    return z_inject, r2


# ---------------------------------------------------------------------------
# Injection hook
# ---------------------------------------------------------------------------

class ResidualHook:
    """Replace the pair output z of a trunk block with a pre-computed tensor."""

    def __init__(self, z_residual: torch.Tensor) -> None:
        self.z_residual = z_residual   # (L, L, D) — will be cast on call

    def __call__(self, _module, _args, output):
        s, z = output
        return (s, self.z_residual[None].to(device=z.device, dtype=z.dtype))


def run_ablated(
    tokenizer,
    model,
    sequence: str,
    device: torch.device,
    layer: int,
    z_residual: torch.Tensor,   # (L, L, D)
) -> np.ndarray:
    """Run ESMFold with z_residual injected at `layer`; return Ca (L, 3)."""
    ids = tokenizer(
        [sequence], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    hook = ResidualHook(z_residual)
    handle = model.trunk.blocks[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(ids)
    finally:
        handle.remove()

    return out["positions"][-1, 0, :, 1, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = choose_device(args.device)

    print(f"Sequence : {args.sequence}")
    print(f"Length   : {len(args.sequence)}")

    tokenizer, model = load_esmfold(device)

    print("\nBaseline forward pass …")
    ca_baseline, z_all = run_baseline(tokenizer, model, args.sequence, device)
    L, pair_dim = z_all.shape[1], z_all.shape[-1]
    print(f"  z_all shape : {z_all.shape}  (48 layers × {L}×{L}×{pair_dim})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "ca_baseline.npy", ca_baseline)

    # ── Per-layer ablation ──────────────────────────────────────────────────
    mode_label = "z_hat only (dual control)" if args.inject_zhat else "z_residual (ablation)"
    print(f"\nMode: {mode_label}")
    print(f"\n{'layer':>6}  {'probe_r2':>9}  {'RMSD(A)':>9}  note")
    print("─" * 42)

    results: list[dict] = []
    for layer in range(47, -1, -1):
        probe_path = args.probe_run_dir / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists():
            print(f"{layer:6d}  {'—':>9}  {'—':>9}  no probe")
            continue

        probe = load_probe(probe_path, pair_dim=pair_dim,
                           hidden_dim=args.probe_hidden_dim, device=device)

        # Split z into probe-explained (z_hat) and residual; inject one half
        z_inject, r2 = compute_probe_split(z_all[layer], probe, device,
                                           inject_zhat=args.inject_zhat)

        # Re-run ESMFold with the chosen half at this layer
        ca_ablated = run_ablated(tokenizer, model, args.sequence, device,
                                 layer, z_inject)

        rmsd = ca_rmsd(ca_baseline, ca_ablated)

        note = ""
        if r2 < 0.2:
            note = "low r2 — probe barely captures z"
        elif rmsd > 2.0:
            note = "*** HIGH RMSD — probe subspace causally important"
        elif rmsd < 0.3:
            note = "model ignores probe subspace"

        print(f"{layer:6d}  {r2:9.3f}  {rmsd:9.3f}  {note}")

        np.save(args.output_dir / f"ca_ablated_layer_{layer:02d}.npy", ca_ablated)
        results.append({
            "layer":            layer,
            "probe_r2":         r2,
            "rmsd_vs_baseline": rmsd,
            "mode":             "zhat" if args.inject_zhat else "residual",
        })

    # ── Summary ─────────────────────────────────────────────────────────────
    out_path = args.output_dir / "summary.json"
    out_path.write_text(json.dumps(
        {"sequence": args.sequence, "results": results}, indent=2
    ))
    print(f"\nSaved to {args.output_dir}/")

    # Quick interpretation
    if results:
        peak = max(results, key=lambda x: x["rmsd_vs_baseline"])
        high_r2 = [r for r in results if r["probe_r2"] > 0.5]
        if high_r2:
            peak_high = max(high_r2, key=lambda x: x["rmsd_vs_baseline"])
            print(f"\nAmong layers with probe_r2 > 0.5:")
            print(f"  Peak RMSD = {peak_high['rmsd_vs_baseline']:.3f} A at layer {peak_high['layer']}")
        print(f"Overall peak RMSD = {peak['rmsd_vs_baseline']:.3f} A at layer {peak['layer']}")


if __name__ == "__main__":
    main()
