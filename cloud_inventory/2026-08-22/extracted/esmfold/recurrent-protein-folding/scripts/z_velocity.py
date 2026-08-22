#!/usr/bin/env python3
"""Per-protein z-velocity curve: ||z_{t+1} − z_t||_F at every layer.

Tests the convexity hypothesis: T1173 only responded to layer-20 perturbation
because layer 20 happens to be a high-velocity point on its trajectory.
Failures had already settled into local pockets by then, so their response
was small at layer 20 but would be larger if perturbed during their own
high-velocity windows.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--proteins", nargs="*", default=None)
    args = ap.parse_args()

    rows = []
    bench_dirs = sorted(args.runs_root.glob("benchmark_*/"))
    for d in bench_dirs:
        name = d.name.replace("benchmark_", "")
        if args.proteins and name not in args.proteins:
            continue
        cache_p = d / "esmfold_activations.pt"
        if not cache_p.exists():
            continue
        try:
            cache = torch.load(cache_p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [{name}] load error: {e}")
            continue
        z = cache["z"].squeeze(1)  # (48, L, L, 128)
        # Compute velocity at each layer (both raw and normalized)
        for t in range(z.shape[0] - 1):
            zt = z[t].float().flatten()
            d = z[t + 1].float().flatten() - zt
            norm = float(d.norm())
            zt_norm = float(zt.norm())
            rel = norm / max(zt_norm, 1e-9)            # fractional change
            n_elems = float(d.numel())
            per_elem = norm / max(n_elems**0.5, 1e-9)  # RMS-per-element
            rows.append({"name": name, "layer": t + 1,
                          "z_velocity": norm,
                          "z_velocity_rel": rel,
                          "z_velocity_per_elem": per_elem})
        print(f"  [{name}] computed velocity over {z.shape[0]-1} layers")
        del cache, z

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
