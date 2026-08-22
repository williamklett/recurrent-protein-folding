#!/usr/bin/env python3
"""Per-protein distogram-space velocity: mean |D_{t+1} - D_t| in Å between
adjacent layers, computed from each protein's MDS PDB stack.

This is the right "what changes between layers" metric — z magnitude is
arbitrary, but a 0.3 Å change in the expected distance matrix is interpretable.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np


def parse_mds_pdb(path: Path) -> list[np.ndarray]:
    """Return list of (L, 3) Ca arrays — one per MODEL."""
    models, cur = [], None
    with path.open() as f:
        for ln in f:
            if ln.startswith("MODEL"):
                cur = []
            elif ln.startswith("ENDMDL") and cur is not None:
                models.append(np.array(cur))
                cur = None
            elif ln.startswith("ATOM") and cur is not None:
                cur.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return models


def pdist_upper(c: np.ndarray) -> np.ndarray:
    diff = c[:, None, :] - c[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/partial_benchmark"))
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--proteins", nargs="*", default=None)
    args = ap.parse_args()

    rows = []
    for d in sorted(args.root.glob("*_analysis")):
        name = d.name.replace("_analysis", "")
        if args.proteins and name not in args.proteins:
            continue
        pdb = d / "mds_structure_video_fixed/distogram_mds_models.pdb"
        if not pdb.exists():
            continue
        models = parse_mds_pdb(pdb)
        if len(models) < 2:
            continue
        Ds = [pdist_upper(c) for c in models]
        for t in range(len(Ds) - 1):
            mae = float(np.abs(Ds[t + 1] - Ds[t]).mean())
            rmse = float(np.sqrt(((Ds[t + 1] - Ds[t]) ** 2).mean()))
            rows.append({"name": name, "layer": t + 1, "dist_mae": mae, "dist_rmse": rmse})
        print(f"  {name}: {len(models)} layers")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
