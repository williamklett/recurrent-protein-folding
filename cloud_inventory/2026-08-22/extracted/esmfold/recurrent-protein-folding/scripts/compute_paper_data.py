#!/usr/bin/env python3
"""Compute and cache per-layer geometric quantities for every protein with cached
ESMFold activations, so all paper figures can be plotted locally from the cache.

For each protein and each trunk layer t we decode the cw30 probe's expected
distance matrix E[d_t] and record:
  - lj         : LJ energy / residue of E[d_t]        (|i-j|>SEQ_SEP)
  - stress     : classical-MDS 3D embedding stress (Å)
  - tri_viol   : triangle-inequality violation rate (random triples)
  - corr_final : Pearson corr of upper-tri E[d_t] vs E[d_final]  (convergence)

Outputs:
  runs/paper_data/<protein>.npz      per-layer arrays
  runs/paper_data/summary.json       scalars (tm, energy descent, violation max, ...)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import (
    ESMFOLD_DISTO_BINS, distogram_bin_centers)

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"
OUT_DIR    = RUNS_DIR / "paper_data"

SIGMA, EPS, SEQ_SEP = 6.8, 1.0, 3
N_TRIPLES = 5000

def available_layers():
    """Layer indices that actually have a trained cw30 probe (belief-replay set
    is layers 01..46; there is no layer_00)."""
    layers = sorted(int(d.name.split("_")[1])
                    for d in PROBES_DIR.glob("layer_*")
                    if (d / "probe.pt").exists())
    return layers


# ── geometry ──────────────────────────────────────────────────────────────────
def lj_from_dist(D, L):
    iu = np.triu_indices(L, k=SEQ_SEP + 1)
    dd = np.clip(D[iu], 0.5, None)
    r  = SIGMA / dd
    return float((4 * EPS * (r ** 12 - r ** 6)).sum() / L)


def classical_mds(D):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ (D ** 2) @ H
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vecs[:, :3] * np.sqrt(np.maximum(vals[:3], 0.0))[None, :]


def mds_stress(D):
    X = classical_mds(D)
    emb = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    iu = np.triu_indices(D.shape[0], 1)
    return float(np.sqrt(np.mean((emb[iu] - D[iu]) ** 2)))


def tri_viol(D, rng):
    L = D.shape[0]
    idx = rng.integers(0, L, size=(N_TRIPLES, 3))
    m = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    i, j, k = idx[m, 0], idx[m, 1], idx[m, 2]
    v = (D[i, k] > D[i, j] + D[j, k]) | (D[i, j] > D[i, k] + D[k, j]) | (D[j, k] > D[j, i] + D[i, k])
    return float(v.mean())


def corr(a, b):
    iu = np.triu_indices(a.shape[0], 1)
    x, y = a[iu], b[iu]
    return float(np.corrcoef(x, y)[0, 1])


# ── probe ─────────────────────────────────────────────────────────────────────
def load_probe(layer, device):
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS,
                                 hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(PROBES_DIR / f"layer_{layer:02d}" / "probe.pt",
                                  map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def decode(z_layer, probe, L, ctr, i_t, j_t, device):
    with torch.no_grad():
        logits = probe.encode(z_layer.to(device).reshape(L * L, 128), i_t, j_t, L)
        d = (F.softmax(logits, dim=-1) * ctr).sum(-1).reshape(L, L)
    d = 0.5 * (d + d.T); d.fill_diagonal_(0.0)
    return d.cpu().numpy()


def tm_lookup():
    tm = {}
    csv = RUNS_DIR / "lj_energy_results_full.csv"
    if csv.exists():
        for line in csv.read_text().splitlines()[1:]:
            parts = line.split(",")
            tm[parts[0]] = float(parts[1])
    return tm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctr = distogram_bin_centers(device, torch.float32).reshape(1, -1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    tms = tm_lookup()

    # cache probes once (shared across proteins)
    layers = available_layers()
    probes = {t: load_probe(t, device) for t in layers}
    print(f"device {device}  loaded {len(probes)} probes (layers "
          f"{layers[0]}..{layers[-1]})")

    acts = sorted(RUNS_DIR.glob("*/esmfold_activations.pt"))
    print(f"found {len(acts)} proteins with activations")

    summary = {}
    for ai, act_p in enumerate(acts):
        prot = act_p.parent.name
        try:
            cache = torch.load(act_p, map_location="cpu", weights_only=False)
            z_all = cache["z"].squeeze(1).float()
        except Exception as e:
            print(f"  [skip] {prot}: {e}")
            continue
        L = z_all.shape[1]
        if L < 4:
            continue
        i_idx, j_idx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        i_t, j_t = i_idx.flatten().to(device), j_idx.flatten().to(device)

        dists = [decode(z_all[t], probes[t], L, ctr, i_t, j_t, device)
                 for t in layers]
        D_final = dists[-1]

        lj     = np.array([lj_from_dist(D, L)   for D in dists])
        stress = np.array([mds_stress(D)        for D in dists])
        viol   = np.array([tri_viol(D, rng)     for D in dists])
        cfin   = np.array([corr(D, D_final)     for D in dists])

        np.savez(OUT_DIR / f"{prot}.npz", lj=lj, stress=stress,
                 tri_viol=viol, corr_final=cfin, L=L,
                 layers=np.array(layers))
        summary[prot] = {
            "tm": tms.get(prot, None), "L": L,
            "E_first": float(lj[0]), "E_final": float(lj[-1]),
            "delta_E": float(lj[-1] - lj[0]), "min_E": float(lj.min()),
            "stress_min": float(stress.min()), "stress_max": float(stress.max()),
            "tri_viol_max": float(viol.max()), "tri_viol_mean": float(viol.mean()),
        }
        print(f"[{ai+1}/{len(acts)}] {prot:32s} L={L:4d}  "
              f"dE={lj[-1]-lj[0]:+.2f}  stress[{stress.min():.1f},{stress.max():.1f}]  "
              f"viol_max={viol.max():.4f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # global headline numbers (for paper verification)
    allv = np.array([s["tri_viol_max"] for s in summary.values()])
    print("\n=== GLOBAL ===")
    print(f"proteins: {len(summary)}")
    print(f"triangle violation: max over all proteins/layers = {allv.max():.6f}")
    print(f"stress band: [{min(s['stress_min'] for s in summary.values()):.2f}, "
          f"{max(s['stress_max'] for s in summary.values()):.2f}] Å")
    print(f"wrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
