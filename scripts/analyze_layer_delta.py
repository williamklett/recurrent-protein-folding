#!/usr/bin/env python3
"""
Per-layer z-difference analysis.

For two proteins A and B, computes the mean pairwise L2 distance between
their pair activations at each trunk block:

    delta_z[T] = mean_{i,j} ||z_A[T,i,j] - z_B[T,i,j]||_2

Then cross-references with gap_closed from a prior donor-z experiment.

Key question: is delta_z flat across layers (all representations similar →
consistency loss killed layer-specific signal) or structured (layer-specific
encoding is real)?

If gap_closed correlates with delta_z across layers → propagation is driven
by layer-specific z-difference, experiment is informative.
If gap_closed is high everywhere and delta_z is flat → we're not showing
layer-specific causality, just that any layer's z encodes the mutation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np
import torch

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-a", type=str, required=True)
    p.add_argument("--seq-b", type=str, required=True)
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--donor-z-summary", type=Path, default=None,
                   help="summary.json from a prior donor_z_propagation run "
                        "for the same pair, to correlate gap_closed with delta_z")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def choose_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def capture_z_all(tokenizer, model, sequence: str, device: torch.device,
                  label: str) -> torch.Tensor:
    import time
    ids = tokenizer([sequence], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    capture = ESMFoldTrunkCapture(model)
    t0 = time.time()
    with capture.capturing(), torch.no_grad():
        model(ids)
    print(f"  [{label}] captured in {time.time()-t0:.1f}s")
    return capture.stacked_z().squeeze(1).float().cpu()   # (48, L, L, 128)


def main() -> None:
    args   = parse_args()
    device = choose_device(args.device)

    assert len(args.seq_a) == len(args.seq_b), "Sequences must be same length"
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(args.seq_a, args.seq_b)) if a != b]
    print(f"Differences: " + ", ".join(f"pos{i} {a}->{b}" for i, a, b in diffs))

    from transformers import AutoTokenizer, EsmForProteinFolding
    print("Loading ESMFold …")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                  low_cpu_mem_usage=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()

    print("Capturing z_all for both sequences …")
    z_a = capture_z_all(tokenizer, model, args.seq_a, device, args.label_a)
    z_b = capture_z_all(tokenizer, model, args.seq_b, device, args.label_b)
    # z: (48, L, L, 128)

    # ── Per-layer delta_z ─────────────────────────────────────────────────
    # Mean over all (i,j) pairs of the L2 norm of the difference vector
    diff   = z_a - z_b                            # (48, L, L, 128)
    # Frobenius per pair, then mean over pairs
    delta_z     = diff.norm(dim=-1).mean(dim=(-1, -2))   # (48,)
    # Relative: divide by mean norm of z_a at that layer
    z_a_norm    = z_a.norm(dim=-1).mean(dim=(-1, -2))    # (48,)
    delta_z_rel = delta_z / z_a_norm.clamp(min=1e-8)     # (48,)

    delta_z_np     = delta_z.numpy()
    delta_z_rel_np = delta_z_rel.numpy()

    # ── Load gap_closed if provided ───────────────────────────────────────
    gap_closed_map: dict[int, float] = {}
    if args.donor_z_summary and args.donor_z_summary.exists():
        summary = json.loads(args.donor_z_summary.read_text())
        gap_closed_map = {r["layer"]: r["gap_closed"] for r in summary["results"]}

    # ── Report ────────────────────────────────────────────────────────────
    has_gc = bool(gap_closed_map)
    header = f"{'layer':>6}  {'delta_z':>9}  {'rel_delta':>10}"
    if has_gc:
        header += f"  {'gap_closed':>11}"
    print(f"\n{header}")
    print("─" * (len(header) + 2))

    rows = []
    for layer in range(47, -1, -1):
        dz  = float(delta_z_np[layer])
        dzr = float(delta_z_rel_np[layer])
        gc  = gap_closed_map.get(layer, None)
        line = f"{layer:6d}  {dz:9.4f}  {dzr:10.4f}"
        if has_gc:
            line += f"  {gc:11.3f}" if gc is not None else f"  {'—':>11}"
        print(line)
        rows.append({"layer": layer, "delta_z": dz, "delta_z_rel": dzr,
                     "gap_closed": gc})

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\ndelta_z across layers:")
    print(f"  mean = {delta_z_np.mean():.4f}")
    print(f"  std  = {delta_z_np.std():.4f}")
    print(f"  CV   = {delta_z_np.std()/delta_z_np.mean():.3f}  "
          f"(coefficient of variation; low → flat → representations too similar)")
    print(f"  min  = {delta_z_np.min():.4f} at layer {int(delta_z_np.argmin())}")
    print(f"  max  = {delta_z_np.max():.4f} at layer {int(delta_z_np.argmax())}")

    if has_gc:
        gc_arr = np.array([gap_closed_map.get(l, np.nan) for l in range(48)])
        valid  = ~np.isnan(gc_arr)
        corr   = float(np.corrcoef(delta_z_np[valid], gc_arr[valid])[0, 1])
        print(f"\nPearson r(delta_z, gap_closed) = {corr:.3f}")
        if abs(corr) > 0.5:
            print("  Strong correlation → propagation is driven by layer-specific Δz")
        elif abs(corr) < 0.2:
            print("  Weak correlation → gap_closed doesn't track Δz; "
                  "uniform propagation regardless of how different z is at that layer")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "seq_a": args.seq_a, "seq_b": args.seq_b,
            "label_a": args.label_a, "label_b": args.label_b,
            "diffs": diffs,
            "results": rows,
        }, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
