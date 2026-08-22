#!/usr/bin/env python3
"""
Donor-z propagation test.

Two similar proteins A and B (same length, differ in ≥1 residue).
For each trunk block T, swap z_B[T] into protein A's forward pass and measure
whether the output structure shifts toward B.

Both z tensors come from real ESMFold runs, so they are on-manifold by
construction — no probe bottleneck, no reconstruction loss.

Key metric per layer
--------------------
  fraction_gap_closed = 1 - RMSD(steered, Ca_B) / RMSD(Ca_A, Ca_B)

  > 0  : output moved toward B  (z_T encodes the structural difference)
  ≈ 0  : no effect              (z_T at this layer doesn't drive the difference)
  < 0  : moved away from B      (injection is counterproductive at this layer)

Examples
--------
  # WT vs Y30F (conservative: tyrosine→phenylalanine, lose one H-bond donor)
  python scripts/donor_z_propagation.py \\
      --seq-a GECEQCFSDGGDCTTCFNNGTGPCANCLAGYPAGCSNSDCTAFLSQCYGGC \\
      --seq-b GECEQCFSDGGDCTTCFNNGTGPCANCLAGFPAGCSNSDCTAFLSQCYGGC \\
      --label-a WT --label-b Y30F \\
      --output-dir runs/donor_z_Y30F

  # WT vs C2A (disruptive: removes first cysteine, breaks one disulfide)
  python scripts/donor_z_propagation.py \\
      --seq-a GECEQCFSDGGDCTTCFNNGTGPCANCLAGYPAGCSNSDCTAFLSQCYGGC \\
      --seq-b GEAEQCFSDGGDCTTCFNNGTGPCANCLAGYPAGCSNSDCTAFLSQCYGGC \\
      --label-a WT --label-b C2A \\
      --output-dir runs/donor_z_C2A
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

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-a", type=str, required=True,
                   help="Protein A amino-acid sequence (baseline / wildtype)")
    p.add_argument("--seq-b", type=str, required=True,
                   help="Protein B amino-acid sequence (target / mutant). "
                        "Must be same length as seq-a.")
    p.add_argument("--label-a", type=str, default="A")
    p.add_argument("--label-b", type=str, default="B")
    p.add_argument("--output-dir", type=Path, default=Path("runs/donor_z"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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


def forward_and_capture(
    tokenizer, model, sequence: str, device: torch.device, label: str
) -> tuple[np.ndarray, torch.Tensor]:
    """Run ESMFold, return (Ca coords (L,3), z_all (48,L,L,128))."""
    ids = tokenizer([sequence], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    capture = ESMFoldTrunkCapture(model)
    t0 = time.time()
    with capture.capturing(), torch.no_grad():
        out = model(ids)
    print(f"  [{label}] forward: {time.time()-t0:.1f}s")
    ca    = out["positions"][-1, 0, :, 1, :].cpu().numpy()   # (L, 3)
    z_all = capture.stacked_z().squeeze(1).float()            # (48, L, L, 128)
    return ca, z_all


# ---------------------------------------------------------------------------
# Donor-z injection
# ---------------------------------------------------------------------------

class DonorHook:
    """Replace z output of a trunk block with a donor tensor."""
    def __init__(self, z_donor: torch.Tensor) -> None:
        self.z_donor = z_donor   # (L, L, D)

    def __call__(self, _module, _args, output):
        s, z = output
        return (s, self.z_donor[None].to(device=z.device, dtype=z.dtype))


def run_with_donor_z(
    tokenizer, model, sequence: str, device: torch.device,
    layer: int, z_donor: torch.Tensor,
) -> np.ndarray:
    """Run ESMFold on `sequence` with z_donor injected at `layer`. Returns Ca (L,3)."""
    ids = tokenizer([sequence], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    hook   = DonorHook(z_donor)
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

    if len(args.seq_a) != len(args.seq_b):
        raise ValueError(
            f"Sequences must be the same length: "
            f"|A|={len(args.seq_a)}, |B|={len(args.seq_b)}"
        )

    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(args.seq_a, args.seq_b)) if a != b]
    print(f"Protein A ({args.label_a}): {args.seq_a}")
    print(f"Protein B ({args.label_b}): {args.seq_b}")
    print(f"Length: {len(args.seq_a)}")
    print(f"Differences ({len(diffs)}): " +
          ", ".join(f"pos{i} {a}->{b}" for i, a, b in diffs))

    tokenizer, model = load_esmfold(device)

    # ── Baseline forward passes ────────────────────────────────────────────
    print(f"\nBaseline forward passes …")
    ca_a, z_all_a = forward_and_capture(tokenizer, model, args.seq_a, device, args.label_a)
    ca_b, z_all_b = forward_and_capture(tokenizer, model, args.seq_b, device, args.label_b)

    baseline_rmsd = ca_rmsd(ca_a, ca_b)
    print(f"\nRMSD({args.label_a}, {args.label_b}) = {baseline_rmsd:.3f} Å  (this is the gap to close)")

    if baseline_rmsd < 0.01:
        print("WARNING: sequences produce identical structures — nothing to propagate.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"ca_{args.label_a}.npy", ca_a)
    np.save(args.output_dir / f"ca_{args.label_b}.npy", ca_b)

    # ── Per-layer donor injection ──────────────────────────────────────────
    print(f"\n{'layer':>6}  {'gap_closed':>11}  {'RMSD→A':>8}  {'RMSD→B':>8}  note")
    print("─" * 56)

    results: list[dict] = []
    for layer in range(47, -1, -1):
        z_donor = z_all_b[layer]   # (L, L, 128) from protein B

        ca_steered = run_with_donor_z(
            tokenizer, model, args.seq_a, device, layer, z_donor
        )

        rmsd_to_a = ca_rmsd(ca_steered, ca_a)
        rmsd_to_b = ca_rmsd(ca_steered, ca_b)
        gap_closed = 1.0 - rmsd_to_b / baseline_rmsd

        note = ""
        if gap_closed > 0.5:
            note = "*** strong propagation"
        elif gap_closed > 0.2:
            note = "* partial propagation"
        elif gap_closed < -0.1:
            note = "  (wrong direction)"

        print(f"{layer:6d}  {gap_closed:11.3f}  {rmsd_to_a:8.3f}  {rmsd_to_b:8.3f}  {note}")

        results.append({
            "layer":       layer,
            "gap_closed":  gap_closed,
            "rmsd_to_a":   rmsd_to_a,
            "rmsd_to_b":   rmsd_to_b,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    out_path = args.output_dir / "summary.json"
    out_path.write_text(json.dumps({
        "seq_a":        args.seq_a,
        "seq_b":        args.seq_b,
        "label_a":      args.label_a,
        "label_b":      args.label_b,
        "diffs":        diffs,
        "baseline_rmsd": baseline_rmsd,
        "results":      results,
    }, indent=2))

    best = max(results, key=lambda r: r["gap_closed"])
    print(f"\nBaseline RMSD (A vs B): {baseline_rmsd:.3f} Å")
    print(f"Best propagation: gap_closed={best['gap_closed']:.3f} at layer {best['layer']}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
