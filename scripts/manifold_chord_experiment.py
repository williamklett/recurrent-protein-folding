#!/usr/bin/env python3
"""Geodesic-vs-chord experiment: does the trunk follow a curved energy manifold?

Take two ON-MANIFOLD trajectory points — the trunk's shapes at block START and
block END (both low-energy configurations the trunk actually visits). Then:

  TRUNK PATH : the trunk's actual shapes at blocks START, START+1, …, END.
  CHORD      : the straight line in 3D COORDINATE space between the aligned
               endpoints, sampled at many points.

Energy is computed from realized 3D coordinates at every point, so each
intermediate is a valid conformation (no triangle-violation artifacts).

Claim: the straight coordinate chord between two on-manifold points spikes in
energy (it cuts through the high-energy interior), while the trunk's actual path
stays low — i.e. the manifold is curved, the trunk follows the curve, the chord
tunnels through.

Two referee concerns this design answers:
  1. Endpoints are both on-manifold low-energy states (not the disordered layer 1),
     so the contrast is fair.
  2. We interpolate COORDINATES (every intermediate is embeddable), not distance
     matrices (whose linear interpolation can be geometrically impossible). We
     additionally report the triangle-violation rate of the distance-matrix
     interpolant midpoint to show *why* coordinate interpolation is necessary.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"

SIGMA   = 6.8
EPS     = 1.0
SEQ_SEP = 3

PROTEIN_ACTS = {
    "8fmi":    "benchmark_8fmi/esmfold_activations.pt",
    "T1137s4": "benchmark_T1137s4/esmfold_activations.pt",
    "ubq":     "benchmark_ubq/esmfold_activations.pt",
}


# ── geometry helpers ────────────────────────────────────────────────────────

def classical_mds(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ (D ** 2) @ H
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    coords = vecs[:, :3] * np.sqrt(np.maximum(vals[:3], 0.0))[None, :]
    return coords - coords.mean(0, keepdims=True)


def smacof(D: np.ndarray, init: np.ndarray, n_iter: int = 300,
           tol: float = 1e-5) -> np.ndarray:
    """Metric-MDS stress majorization (Guttman transform). Refines `init` to a
    low-stress 3D embedding of the target distance matrix D. Classical MDS gives
    a high-stress one-shot embedding whose collapsed pairs blow up the LJ energy;
    SMACOF drives the realized inter-residue distances onto D so the energy
    computed from coordinates is physical."""
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


def embed(D: np.ndarray) -> np.ndarray:
    return smacof(D, classical_mds(D))


def canonical(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    signs = np.sign((x @ vt.T).mean(0)); signs[signs == 0] = 1.0
    vt = (vt.T * signs).T
    if np.linalg.det(vt) < 0:
        vt[-1] *= -1
    return x @ vt.T


def kabsch(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reflection-safe rotation of x onto ref."""
    x0 = x - x.mean(0, keepdims=True)
    y0 = ref - ref.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(x0.T @ y0)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1.0, 1.0, d]) @ vt
    return x0 @ R


def stress(coords: np.ndarray, D: np.ndarray) -> float:
    emb = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    iu = np.triu_indices(D.shape[0], 1)
    return float(np.sqrt(np.mean((emb[iu] - D[iu]) ** 2)))


def lj_from_coords(coords: np.ndarray, L: int) -> float:
    d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    iu = np.triu_indices(L, k=SEQ_SEP + 1)        # |i-j| > SEQ_SEP
    dd = np.clip(d[iu], 0.5, None)
    r  = SIGMA / dd
    return float((4 * EPS * (r ** 12 - r ** 6)).sum() / L)


def lj_from_dist(D: np.ndarray, L: int) -> float:
    iu = np.triu_indices(L, k=SEQ_SEP + 1)
    dd = np.clip(D[iu], 0.5, None)
    r  = SIGMA / dd
    return float((4 * EPS * (r ** 12 - r ** 6)).sum() / L)


def triangle_violation_rate(D: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    L = D.shape[0]
    rng = np.random.default_rng(seed)
    i, j, k = (rng.integers(0, L, n) for _ in range(3))
    ok = (i != j) & (j != k) & (i != k)
    i, j, k = i[ok], j[ok], k[ok]
    viol = D[i, k] > D[i, j] + D[j, k] + 1e-6
    return float(viol.mean())


# ── probe decode ─────────────────────────────────────────────────────────────

def load_probe(layer: int, pd: int, device: torch.device) -> PairDistogramBottleneck:
    pr = PairDistogramBottleneck(pair_dim=pd, bins=ESMFOLD_DISTO_BINS,
                                 hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(
        PROBES_DIR / f"layer_{layer:02d}" / "probe.pt",
        map_location=device, weights_only=True))
    for p in pr.parameters():
        p.requires_grad_(False)
    return pr


def decode_dist(z_layer, layer, L, pd, i_t, j_t, ctr, device) -> np.ndarray:
    pr = load_probe(layer, pd, device)
    with torch.no_grad():
        logits = pr.encode(z_layer.to(device).reshape(L * L, pd), i_t, j_t, L)
        d = (F.softmax(logits, dim=-1) * ctr).sum(-1).reshape(L, L)
    d = 0.5 * (d + d.T)
    d.fill_diagonal_(0.0)
    return d.cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--end",   type=int, default=46)
    ap.add_argument("--n-interp", type=int, default=60)
    ap.add_argument("--proteins", nargs="+", default=["8fmi", "T1137s4", "ubq"])
    args = ap.parse_args()

    START, END = args.start, args.end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctr = distogram_bin_centers(device, torch.float32).reshape(1, -1)
    print(f"device {device}  start={START}  end={END}")

    all_results = {}

    for prot in args.proteins:
        print(f"\n{'='*55}\n{prot}")
        cache = torch.load(RUNS_DIR / PROTEIN_ACTS[prot],
                           map_location="cpu", weights_only=False)
        z_all = cache["z"].squeeze(1).float()
        L  = z_all.shape[1]
        pd = z_all.shape[-1]

        i_idx, j_idx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
        i_t = i_idx.flatten().to(device)
        j_t = j_idx.flatten().to(device)

        # decode + MDS for every block on the trunk path
        blocks = list(range(START, END + 1))
        dist   = {t: decode_dist(z_all[t], t, L, pd, i_t, j_t, ctr, device) for t in blocks}
        coords = {t: embed(dist[t]) for t in blocks}

        # align everything to END frame (LJ is rotation-invariant, but the chord
        # needs both endpoints in a shared frame)
        coords[END] = canonical(coords[END])
        for t in blocks:
            if t != END:
                coords[t] = kabsch(coords[t], coords[END])

        s_start = stress(coords[START], dist[START])
        s_end   = stress(coords[END],   dist[END])
        print(f"  MDS stress: block {START}={s_start:.2f}Å  block {END}={s_end:.2f}Å")

        # endpoint separation (both already in the END frame): tells flat cases
        # ("endpoints too similar — nothing to morph through") apart from real nulls
        end_rmsd = float(np.sqrt(((coords[START] - coords[END]) ** 2).sum(1).mean()))
        # also report it relative to each structure's own radius of gyration
        rg = float(np.sqrt((coords[END] ** 2).sum(1).mean()))
        print(f"  Endpoint RMSD (block {START} vs {END}): {end_rmsd:.2f}Å   (Rg={rg:.2f}Å, RMSD/Rg={end_rmsd/rg:.2f})")

        # ── trunk path energy (from realized coords) ─────────────────────────
        trunk_prog = np.array([(t - START) / (END - START) for t in blocks])
        trunk_E    = np.array([lj_from_coords(coords[t], L) for t in blocks])

        # ── coordinate chord energy ──────────────────────────────────────────
        chord_prog = np.linspace(0.0, 1.0, args.n_interp)
        chord_E    = []
        for s in chord_prog:
            c = (1.0 - s) * coords[START] + s * coords[END]
            chord_E.append(lj_from_coords(c, L))
        chord_E = np.array(chord_E)

        # ── diagnostic: distance-matrix chord (the naive version) ────────────
        dist_chord_E = []
        for s in chord_prog:
            D_s = (1.0 - s) * dist[START] + s * dist[END]
            dist_chord_E.append(lj_from_dist(D_s, L))
        dist_chord_E = np.array(dist_chord_E)
        D_mid = 0.5 * (dist[START] + dist[END])
        tri_end   = triangle_violation_rate(dist[END])
        tri_mid   = triangle_violation_rate(D_mid)

        print(f"  Endpoint LJ:  block {START}={trunk_E[0]:+.3f}  block {END}={trunk_E[-1]:+.3f}")
        print(f"  Trunk path  peak LJ: {trunk_E.max():+.3f}  (over interior)")
        print(f"  Coord chord peak LJ: {chord_E.max():+.3f}")
        print(f"  → chord rises {chord_E.max() - max(trunk_E[0], trunk_E[-1]):+.3f} above the higher endpoint")
        print(f"  Triangle viol: endpoint={tri_end*100:.2f}%  dist-chord midpoint={tri_mid*100:.2f}%")
        print(f"  Dist-matrix chord peak LJ: {dist_chord_E.max():+.3f} (contaminated by invalid geometry)")

        all_results[prot] = {
            "trunk_prog": trunk_prog.tolist(),
            "trunk_E":    trunk_E.tolist(),
            "chord_prog": chord_prog.tolist(),
            "chord_E":    chord_E.tolist(),
            "dist_chord_E": dist_chord_E.tolist(),
            "stress_start": s_start, "stress_end": s_end,
            "end_rmsd": end_rmsd, "rg": rg,
            "tri_end": tri_end, "tri_mid": tri_mid,
            "L": L,
        }

    # ── save + plot ───────────────────────────────────────────────────────────
    out_json = RUNS_DIR / f"manifold_chord_{START}_{END}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_json}")

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5), squeeze=False)
    fig.suptitle(
        f"The trunk follows a curved energy manifold; the straight chord tunnels through high energy "
        f"(blocks {START}→{END})",
        fontsize=13)

    for ax, (prot, r) in zip(axes[0], all_results.items()):
        tp = np.array(r["trunk_prog"]); te = np.array(r["trunk_E"])
        cp = np.array(r["chord_prog"]); ce = np.array(r["chord_E"])
        ax.plot(cp, ce, color="#dc2626", lw=2.4,
                label="straight coordinate chord")
        ax.fill_between(cp, ce, te.min(), color="#dc2626", alpha=0.07)
        ax.plot(tp, te, color="#16a34a", lw=2.6, marker="o", ms=3,
                label="trunk's actual path")
        ax.scatter([0, 1], [te[0], te[-1]], color="#0f172a", s=55, zorder=5,
                   label="shared endpoints (on-manifold)")
        ax.set_xlabel(f"path progress  (block {START} → {END})", fontsize=11)
        ax.set_ylabel("LJ energy / residue (from 3D coords)", fontsize=11)
        ax.set_title(prot, fontsize=12)
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_png = RUNS_DIR / f"manifold_chord_{START}_{END}.png"
    fig.savefig(out_png, dpi=160)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
