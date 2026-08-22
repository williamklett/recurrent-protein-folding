#!/usr/bin/env python3
"""Apply general probes to a protein's cached z activations.

For each layer t in 1..46: compute E[d]_t = expected distance from probe_t(z_t).
For layer 47: compute E[d]_47 = expected distance from ESMFold's distogram head
applied to z_47 (this is the canonical "model output" reference).

Then compute corr(E[d]_t, E[d]_47) for each t, plus dist_mae_vs_47.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import (
    ESMFOLD_DISTO_BINS,
    distogram_bin_centers,
)


def expected_distance(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """logits: (..., bins) → expected distance: (..., )"""
    return (F.softmax(logits, dim=-1) * bin_centers).sum(-1)


def apply_probe(probe, z_layer, device):
    L = z_layer.shape[0]
    pair_dim = z_layer.shape[-1]
    flat = z_layer.reshape(-1, pair_dim).to(device)
    i_idx, j_idx = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    with torch.no_grad():
        logits = probe.encode(flat, i_idx.flatten(), j_idx.flatten(), L)
    return logits.reshape(L, L, -1)


def apply_head(head, z_47, device):
    with torch.no_grad():
        z = z_47.to(device).float()
        z_sym = 0.5 * (z + z.transpose(0, 1))
        return head(z_sym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-root", type=Path, required=True,
                    help="directory containing layer_01..layer_46 with probe.pt files")
    ap.add_argument("--protein-dirs-file", type=Path, required=True,
                    help="text file with paths to protein dirs (containing esmfold_activations.pt)")
    ap.add_argument("--out-csv-dir", type=Path, required=True,
                    help="directory where per-protein corr_vs_47.csv files go")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bin_centers = distogram_bin_centers(device, torch.float32)

    # Load ESMFold's distogram head
    print("loading ESMFold distogram head...")
    m = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    head = m.distogram_head.to(device).eval()
    for p in head.parameters(): p.requires_grad_(False)
    del m
    torch.cuda.empty_cache()

    # Load all 46 general probes once
    print("loading 46 general probes...")
    probes = {}
    for layer in range(1, 47):
        p_path = args.probes_root / f"layer_{layer:02d}" / "probe.pt"
        if not p_path.exists():
            raise FileNotFoundError(f"missing {p_path}")
        pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
        pr.load_state_dict(torch.load(p_path, map_location=device, weights_only=True))
        for pp in pr.parameters(): pp.requires_grad_(False)
        probes[layer] = pr
    print(f"  loaded probes for layers {min(probes)}..{max(probes)}")

    args.out_csv_dir.mkdir(parents=True, exist_ok=True)
    protein_dirs = [Path(l.strip()) for l in args.protein_dirs_file.read_text().splitlines() if l.strip()]
    print(f"evaluating {len(protein_dirs)} proteins...")

    for d in protein_dirs:
        cache_p = d / "esmfold_activations.pt"
        if not cache_p.exists():
            print(f"  [{d.name}] skip - no cache")
            continue
        cache = torch.load(cache_p, map_location="cpu", weights_only=False)
        z_full = cache["z"].squeeze(1)
        L = z_full.shape[1]
        del cache
        print(f"  [{d.name}] L={L}")

        # Compute E[d]_47 from ESMFold head
        z_47 = z_full[-1].float().to(device)
        head_logits = apply_head(head, z_47, device)
        ed_47 = expected_distance(head_logits, bin_centers).cpu().numpy()
        del z_47

        # Use upper-triangle (i<j) for pairwise comparisons
        iu = np.triu_indices(L, k=1)
        ed_47_flat = ed_47[iu]

        rows = []
        for layer in range(1, 47):
            z_t = z_full[layer - 1].float()  # z[layer-1] is the input to layer; layer 1 = z[0]
            probe = probes[layer]
            logits_t = apply_probe(probe, z_t, device)
            ed_t = expected_distance(logits_t, bin_centers).cpu().numpy()
            ed_t_flat = ed_t[iu]

            corr = float(np.corrcoef(ed_t_flat, ed_47_flat)[0, 1])
            dist_mae = float(np.abs(ed_t_flat - ed_47_flat).mean())
            rows.append({"layer": layer, "corr_vs_47": corr, "dist_mae_vs_47": dist_mae,
                          "dist_mae_vs_centroid": "",
                          "corr_vs_centroid": ""})
            del z_t, logits_t
            torch.cuda.empty_cache()

        # Add layer 47 (the reference itself) for completeness
        rows.append({"layer": 47, "corr_vs_47": 1.0, "dist_mae_vs_47": 0.0,
                      "dist_mae_vs_centroid": "", "corr_vs_centroid": ""})

        # Compute centroid-relative metrics too
        all_ed = []
        for layer in range(1, 48):
            if layer < 47:
                z_t = z_full[layer - 1].float()
                logits_t = apply_probe(probes[layer], z_t, device)
                ed = expected_distance(logits_t, bin_centers).cpu().numpy()[iu]
            else:
                ed = ed_47_flat
            all_ed.append(ed)
        all_ed = np.stack(all_ed)  # (47, n_pairs)
        centroid = all_ed.mean(0)
        for r in rows:
            t = r["layer"]
            ed = all_ed[t - 1]
            r["dist_mae_vs_centroid"] = float(np.abs(ed - centroid).mean())
            r["corr_vs_centroid"] = float(np.corrcoef(ed, centroid)[0, 1])

        out_path = args.out_csv_dir / f"{d.name.replace('benchmark_', '').replace('hard_synthetic_miss_', '').replace('synthetic_design_', '')}_layerwise.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"    wrote {out_path.name}")
        del z_full
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
