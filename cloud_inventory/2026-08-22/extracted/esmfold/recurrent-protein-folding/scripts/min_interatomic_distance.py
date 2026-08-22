#!/usr/bin/env python3
"""Cross-trajectory minimum interatomic distance, all 57 proteins.

Potential-free version of the path-vs-chord experiment (scripts/manifold_chord_
experiment.py): the honest geometric metric is the closest approach between two
sequence-distal residues (|i-j|>SEQ_SEP), in Angstrom, NOT the Lennard-Jones spike.

Trajectory spans the WHOLE trunk, blocks START=1 .. END=46.

TRUNK closest approach is read directly from the probe's believed distance matrix
D_t = E[d] at each block --- no 3D embedding, so no MDS-stress artifact.  For a
physically realizable fold no two C-alpha can occupy the same point, so this
believed closest approach staying bounded away from 0 is the statement that the
trunk's path is sterically valid at every step.

The CHORD is the straight line in 3D COORDINATE space between the block-1 and
block-46 structures (you cannot linearly interpolate distance matrices --- the
interpolant violates the triangle inequality and is not embeddable --- so the
direct route must be built in coordinate space).  We SMACOF-embed only the two
endpoints, Kabsch-align them, and report the chord's closest approach as it
sweeps from one to the other.  end_min/stress are reported so degenerate
(near-identical-endpoint) and high-stress (large-L) cases are visible.

Split by fold success (TM>=0.5) vs failure (TM<0.5); TM from the existing
runs/triangle_violation_results.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck

# Inlined from recurrent_folding.train_esmfold_distogram_probes to avoid its
# h5py import (not installed in this env).
ESMFOLD_DISTO_BINS = 64
ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875


def distogram_bin_centers(device, dtype):
    edges = torch.linspace(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX,
                           ESMFOLD_DISTO_BINS + 1, device=device, dtype=dtype)
    return ((edges[:-1] + edges[1:]) * 0.5).view(1, -1)


ROOT = Path.home() / "recurrent-protein-folding"
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR = ROOT / "runs"
TM_CSV = RUNS_DIR / "triangle_violation_results.csv"
SEQ_SEP = 3
START, END = 1, 46
N_INTERP = 200
CLASH = 4.0   # Angstrom; fraction of the chord below this counts as "in steric clash"


# ---- geometry (identical to manifold_chord_experiment.py) -------------------
def classical_mds(D):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ (D ** 2) @ H
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    coords = vecs[:, :3] * np.sqrt(np.maximum(vals[:3], 0.0))[None, :]
    return coords - coords.mean(0, keepdims=True)


def smacof(D, init, n_iter=300, tol=1e-5):
    n = D.shape[0]
    X = init.copy()
    iu = np.triu_indices(n, 1)

    def emb_dist(Y):
        return np.sqrt(np.maximum(((Y[:, None] - Y[None]) ** 2).sum(-1), 1e-12))

    d = emb_dist(X)
    prev = np.sqrt(np.mean((d[iu] - D[iu]) ** 2))
    for _ in range(n_iter):
        d = emb_dist(X)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d > 1e-8, D / d, 0.0)
        B = -ratio
        np.fill_diagonal(B, 0.0)
        B[np.diag_indices(n)] = -B.sum(1)
        X = (B @ X) / n
        X -= X.mean(0, keepdims=True)
        cur = np.sqrt(np.mean((emb_dist(X)[iu] - D[iu]) ** 2))
        if abs(prev - cur) < tol:
            break
        prev = cur
    return X


def embed(D):
    return smacof(D, classical_mds(D))


def canonical(x):
    x = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    signs = np.sign((x @ vt.T).mean(0)); signs[signs == 0] = 1.0
    vt = (vt.T * signs).T
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return x @ vt.T


def kabsch(x, ref):
    x0 = x - x.mean(0, keepdims=True)
    y0 = ref - ref.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(x0.T @ y0)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1.0, 1.0, d]) @ vt
    return x0 @ R


def min_pair_dist(coords, iu):
    d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    return float(d[iu].min())


def stress(coords, D):
    emb = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    iu = np.triu_indices(D.shape[0], 1)
    return float(np.sqrt(np.mean((emb[iu] - D[iu]) ** 2)))


# ---- probe decode -----------------------------------------------------------
def load_probe(layer, pd, device):
    pr = PairDistogramBottleneck(pair_dim=pd, bins=ESMFOLD_DISTO_BINS,
                                 hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(PROBES_DIR / f"layer_{layer:02d}" / "probe.pt",
                                  map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def decode_dist(z_layer, layer, L, pd, i_t, j_t, ctr, device):
    pr = load_probe(layer, pd, device)
    with torch.no_grad():
        logits = pr.encode(z_layer.to(device).reshape(L * L, pd), i_t, j_t, L)
        d = (F.softmax(logits, dim=-1) * ctr).sum(-1).reshape(L, L)
    d = 0.5 * (d + d.T)
    d.fill_diagonal_(0.0)
    return d.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctr = distogram_bin_centers(device, torch.float32).reshape(1, -1)

    tm = {r["name"]: float(r["tm"]) for r in csv.DictReader(open(TM_CSV))}
    names = list(tm)
    print(f"device {device}  N={len(names)} proteins  blocks {START}->{END}", flush=True)

    results = {}
    for k, name in enumerate(names):
        cache = torch.load(RUNS_DIR / name / "esmfold_activations.pt",
                           map_location="cpu", weights_only=False)
        z_all = cache["z"].squeeze(1).float()
        L, pd = z_all.shape[1], z_all.shape[-1]
        ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        i_t, j_t = ii.flatten().to(device), jj.flatten().to(device)
        iu = np.triu_indices(L, k=SEQ_SEP + 1)        # |i-j| > SEQ_SEP

        blocks = list(range(START, END + 1))
        # TRUNK closest approach: read straight from the believed distance matrix
        # at every block (no embedding, so no MDS-stress artifact).
        d_end = None
        d_start = None
        trunk_block_minD = {}
        for t in blocks:
            Dt = decode_dist(z_all[t], t, L, pd, i_t, j_t, ctr, device)
            trunk_block_minD[t] = float(Dt[iu].min())
            if t == START:
                d_start = Dt
            if t == END:
                d_end = Dt
        trunk_minD = min(trunk_block_minD.values())
        end_minD = min(trunk_block_minD[START], trunk_block_minD[END])

        # CHORD: embed only the two endpoints, align, sweep the straight line.
        c_start = embed(d_start)
        c_end = canonical(embed(d_end))
        c_start = kabsch(c_start, c_end)
        end_min_coords = min(min_pair_dist(c_start, iu), min_pair_dist(c_end, iu))

        chord_min_interior = np.inf
        n_clash = 0
        interior = np.linspace(0.0, 1.0, N_INTERP)[1:-1]
        for s in interior:
            c = (1.0 - s) * c_start + s * c_end
            m = min_pair_dist(c, iu)
            chord_min_interior = min(chord_min_interior, m)
            if m < CLASH:
                n_clash += 1
        chord_min_interior = float(chord_min_interior)
        chord_clash_frac = n_clash / len(interior)   # fraction of chord length in clash

        end_rmsd = float(np.sqrt(((c_start - c_end) ** 2).sum(1).mean()))
        rg = float(np.sqrt((c_end ** 2).sum(1).mean()))
        s_start = stress(c_start, d_start)
        s_end = stress(c_end, d_end)

        results[name] = dict(
            tm=tm[name], L=L,
            trunk_minD=trunk_minD, end_minD=end_minD,
            chord_min_interior=chord_min_interior, chord_clash_frac=chord_clash_frac,
            end_min_coords=end_min_coords,
            end_rmsd=end_rmsd, rg=rg, rmsd_over_rg=end_rmsd / rg,
            stress_start=s_start, stress_end=s_end,
        )
        print(f"[{k+1:2d}/{len(names)}] {name:32s} TM={tm[name]:.3f} L={L:3d}  "
              f"trunk_minD={trunk_minD:5.2f}A  chord_min(int)={chord_min_interior:5.2f}A  "
              f"clash_frac={chord_clash_frac:.2f}  rmsd/rg={end_rmsd/rg:.2f}  "
              f"end_c={end_min_coords:4.2f} stress={s_end:.2f}A", flush=True)
        del cache, z_all
        torch.cuda.empty_cache()

    out_json = RUNS_DIR / "min_interatomic_distance_57.json"
    json.dump(results, open(out_json, "w"), indent=2)
    out_csv = RUNS_DIR / "min_interatomic_distance_57.csv"
    cols = ["name", "tm", "L", "trunk_minD", "end_minD", "chord_min_interior",
            "chord_clash_frac", "end_min_coords", "end_rmsd", "rg", "rmsd_over_rg",
            "stress_start", "stress_end"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n, r in results.items():
            w.writerow([n] + [r[c] for c in cols[1:]])

    # ---- summary, split by fold success ------------------------------------
    MORPH = 0.30   # endpoints must actually differ for the chord to be a real
                   # alternative path (rmsd/rg threshold); flat cases are degenerate

    def block(rs, lbl):
        if not rs:
            print(f"  {lbl:22s}  (none)"); return
        tmn = np.array([r["trunk_minD"] for r in rs])
        cmn = np.array([r["chord_min_interior"] for r in rs])
        print(f"  {lbl:22s} N={len(rs):2d}  "
              f"trunk_minD: med={np.median(tmn):4.2f} min={tmn.min():4.2f} "
              f"p10={np.percentile(tmn,10):4.2f}A | "
              f"chord_min: med={np.median(cmn):4.2f} max={cmn.max():4.2f}A")

    allr = list(results.values())
    morph = [r for r in allr if r["rmsd_over_rg"] >= MORPH]
    succ = [r for r in morph if r["tm"] >= 0.5]
    fail = [r for r in morph if r["tm"] < 0.5]
    print(f"\n{'='*72}\nCLOSEST APPROACH (min interatomic distance, |i-j|>{SEQ_SEP}A), "
          f"blocks {START}-{END}")
    print("trunk_minD = trunk's BELIEVED closest approach (probe D, no embedding)")
    print("chord_min  = straight coordinate chord between block-1 and block-46 structures")
    block(allr, "ALL 57")
    print(f"  --- restricted to real morphs (rmsd/rg>={MORPH}; chord is a genuine alt. path) ---")
    block(morph, "MORPHS (all)")
    block(succ, "MORPHS, SUCCESS TM>=0.5")
    block(fail, "MORPHS, FAILURE  TM<0.5")

    at = np.array([r["trunk_minD"] for r in allr])
    mc = np.array([r["chord_min_interior"] for r in morph])
    print(f"\nTrunk's believed geometry never brings two |i-j|>{SEQ_SEP} residues "
          f"closer than {at.min():.2f}A on any of the 57 trajectories (median "
          f"{np.median(at):.2f}A).")
    print(f"On the {len(morph)} real morphs, the straight chord's closest approach is "
          f"median {np.median(mc):.2f}A (worst {mc.max():.2f}A); "
          f"{(mc < 1.0).sum()}/{len(morph)} collapse below 1A.")
    nflat = len(allr) - len(morph)
    print(f"({nflat}/57 excluded as near-identical-endpoint cases, rmsd/rg<{MORPH}: "
          f"folding already essentially complete by block 1's decoded structure --- "
          f"no morph for a chord to short-circuit.)")
    print(f"\nSaved -> {out_json}\nSaved -> {out_csv}")
    print("DONE_MINDIST", flush=True)


if __name__ == "__main__":
    main()
