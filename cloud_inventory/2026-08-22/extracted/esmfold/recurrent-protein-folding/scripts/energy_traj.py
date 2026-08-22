#!/usr/bin/env python3
"""Direct sanity check: what is the actual LJ energy E(ell_t) of the belief at each
layer?  No cosines, no gradients -- just read each layer with its OWN in-distribution
probe and evaluate the same fixed energy E(ell)=sum_pairs LJ(softmax(ell).centers).

If the trunk is 'folding', E should fall across layers.  The belief-GD cosine claimed
early steps are ANTI-descent (energy rising) and late steps descent (energy falling);
this script confirms or refutes that by the energy curve itself.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from recurrent_folding.esmfold_probes import PairDistogramBottleneck

ROOT = Path.home() / "recurrent-protein-folding"
PROBE_DIR = ROOT / "runs" / "general_belief_replay_benchmark20_cw30"
SEQ_SEP = 3
SIGMA, EPS = 6.8, 1.0
BENCH = {"ubq": "benchmark_ubq", "mbn": "benchmark_mbn", "1AO6": "benchmark_1AO6"}


def centers(device):
    s, e, b = 2.3125, 21.6875, 64
    w = (e - s) / b
    return torch.linspace(s + w / 2, e - w / 2, b, device=device)


def load_probe(layer, device):
    p = PROBE_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not p.exists():
        return None
    pr = PairDistogramBottleneck(pair_dim=128, bins=64, hidden_dim=256).to(device)
    pr.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    pr.eval()
    for q in pr.parameters():
        q.requires_grad = False
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", default="ubq,mbn,1AO6")
    ap.add_argument("--S", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)
    probes = {t: load_probe(t, device) for t in range(1, 47)}

    def sym_belief(pr, zmap, it, jt, L):
        lij = pr.encode(zmap[it, jt], it, jt, L)
        lji = pr.encode(zmap[jt, it], jt, it, L)
        return 0.5 * (lij + lji)

    def energy(ell):
        p = F.softmax(ell, -1)
        ed = (p * cen).sum(-1).clamp(min=0.5)
        rr = SIGMA / ed
        return (4 * EPS * (rr ** 12 - rr ** 6))   # per-pair energy (N,)

    # per-protein normalized energy (mean per-pair) so proteins are comparable
    per_layer = {t: [] for t in range(1, 47)}
    meanED = {t: [] for t in range(1, 47)}
    for short in args.proteins.split(","):
        d = ROOT / "runs" / BENCH[short]
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]; L = z_full.shape[2]
        ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
        if len(ii) > args.S:
            sel = rng.choice(len(ii), args.S, replace=False); ii, jj = ii[sel], jj[sel]
        it = torch.as_tensor(ii, device=device); jt = torch.as_tensor(jj, device=device)
        for t in range(1, 47):
            z = z_full[t, 0].to(device, torch.float32)
            ell = sym_belief(probes[t], z, it, jt, L)
            e = energy(ell)
            per_layer[t].append(e.mean().item())
            p = F.softmax(ell, -1)
            meanED[t].append((p * cen).sum(-1).mean().item())
        del z_full, blob
        print(f"  done {short} (L={L})", flush=True)

    print(f"\n{'L':>3} | {'E/pair':>10} | {'meanED(A)':>9} | {'dE':>9}", flush=True)
    prev = None
    for t in range(1, 47):
        e = float(np.mean(per_layer[t]))
        ed = float(np.mean(meanED[t]))
        de = "" if prev is None else f"{e - prev:>9.4f}"
        print(f"{t:>3} | {e:>10.4f} | {ed:>9.3f} | {de:>9}", flush=True)
        prev = e

    def blk(lo, hi):
        es = [float(np.mean(per_layer[t])) for t in range(lo, hi + 1)]
        return es[0], es[-1], es[-1] - es[0]
    print(f"\n{'block':>12} | {'E_start':>8} {'E_end':>8} {'dE':>8}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-46", 32, 46)]:
        a, b, c = blk(lo, hi)
        print(f"{name:>12} | {a:>8.4f} {b:>8.4f} {c:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
