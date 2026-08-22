#!/usr/bin/env python3
"""
Run ESMFold locally on a sequence and print the mean pLDDT.

Mac-friendly: tries MPS first, falls back to CPU.
For small/medium proteins (< 600 residues) this is workable.
"""
from __future__ import annotations
import argparse, time, json, sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", type=str, required=True)
    p.add_argument("--label",    type=str, default="prot")
    p.add_argument("--output",   type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[{args.label}] length = {len(args.sequence)}")

    # Pick device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype  = torch.float32  # MPS doesn't always like fp16
    else:
        device = torch.device("cpu")
        dtype  = torch.float32
    print(f"  device = {device}")

    from transformers import AutoTokenizer, EsmForProteinFolding
    print("  Loading ESMFold (first run downloads ~3 GB) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                  low_cpu_mem_usage=True)
    model = model.to(device).eval()
    print(f"    loaded in {time.time()-t0:.1f}s")

    # Smaller chunks for memory
    if hasattr(model, "trunk"):
        try:
            model.trunk.set_chunk_size(64)
        except Exception:
            pass

    ids = tokenizer([args.sequence], return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)

    print("  Forward pass ...")
    t0 = time.time()
    with torch.no_grad():
        out = model(ids)
    dt = time.time() - t0
    print(f"    forward in {dt:.1f}s")

    # Mean per-residue pLDDT
    # ESMFold's output['plddt']: (B, L, 37) per-atom; we want per-Ca (atom index 1)
    plddt = out["plddt"]
    print(f"    plddt shape = {tuple(plddt.shape)}")
    # Normalize: ESMFold sometimes returns 0-1, sometimes 0-100
    plddt_ca = plddt[0, :, 1].cpu().float()
    mean_plddt = float(plddt_ca.mean())
    # Compress to 0-1 if it looks like a percentage
    if mean_plddt > 1.5:
        mean_plddt = mean_plddt / 100.0
        plddt_ca = plddt_ca / 100.0
    print(f"  Mean pLDDT (Cα): {mean_plddt:.4f}")
    print(f"  Distribution: min={float(plddt_ca.min()):.3f}  "
          f"med={float(plddt_ca.median()):.3f}  max={float(plddt_ca.max()):.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "label": args.label,
            "sequence": args.sequence,
            "length": len(args.sequence),
            "mean_plddt": mean_plddt,
            "plddt_per_ca": plddt_ca.tolist(),
            "wall_seconds": dt,
        }))
        print(f"  Saved {args.output}")


if __name__ == "__main__":
    main()
