#!/usr/bin/env python3
"""Dimension-controlled belief-GD test.  In-distribution readouts only.

ell_t = encode_t(z_t);  step = ell_{t+1} - ell_t;  g = grad_ell E(ell_t).
Energy E = sum_pairs LJ(softmax(ell).centers)  -> per pair the gradient is a SINGLE
direction (only the expected distance matters), so the descent target is ~1-D per pair
and ~62 dims are energy-neutral.

We report the realized alignment NOT against 0 but against a dimension-and-magnitude-
matched NULL: permute which pair's gradient each step is scored against (preserves all
norms, heterogeneity, dimensionality; destroys only the true step<->grad correspondence).

  obs_g   : observed global (magnitude-weighted) cosine
  null_g  : mean global cosine under random pair-permutation of the gradient
  z_g     : (obs - null_mean)/null_std            <- dimension-free effect size
  R2_pp   : mean over pairs of cos^2(step, grad)  (= energy-relevant fraction of step var)
            compare to 1/Deff (random) where Deff ~ 63
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
    ap.add_argument("--damping", type=float, default=1e-2)
    ap.add_argument("--nperm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)
    g_cpu = torch.Generator(device=device)
    probes = {t: load_probe(t, device) for t in range(1, 47)}

    def sym_belief(pr, zmap, it, jt, L):
        lij = pr.encode(zmap[it, jt], it, jt, L)
        lji = pr.encode(zmap[jt, it], jt, it, L)
        return 0.5 * (lij + lji)

    def grad_natural(ell, damping):
        L0 = ell.detach().clone().requires_grad_(True)
        p = F.softmax(L0, -1); ed = (p * cen).sum(-1).clamp(min=0.5)
        rr = SIGMA / ed
        E = (4 * EPS * (rr ** 12 - rr ** 6)).sum()
        g = torch.autograd.grad(E, L0)[0]
        with torch.no_grad():
            pp = F.softmax(L0, -1)
            Fm = torch.diag_embed(pp) - pp[..., :, None] * pp[..., None, :]
            lam = damping * (pp - pp ** 2).mean(-1, keepdim=True)
            Fd = Fm + lam[..., None] * torch.eye(64, device=device)
            x = torch.linalg.solve(Fd, g.unsqueeze(-1)).squeeze(-1)
        return g.detach(), x

    def stats(step, neg_dir, nperm):
        N = step.shape[0]
        sd = step.norm(); nd = neg_dir.norm()
        obs = (step * neg_dir).sum().item() / (sd.item() * nd.item() + 1e-12)
        # per-pair cos^2 (energy-relevant fraction of step variance)
        cpp = F.cosine_similarity(step, neg_dir, dim=-1)
        r2 = (cpp ** 2).mean().item()
        # permutation null on the global cosine (norms unchanged under permutation)
        nums = torch.empty(nperm, device=device)
        denom = sd * nd + 1e-12
        for q in range(nperm):
            perm = torch.randperm(N, device=device, generator=g_cpu)
            nums[q] = (step * neg_dir[perm]).sum() / denom
        return obs, r2, nums.mean().item(), nums.std().item()

    g_cpu.manual_seed(args.seed)
    keys = ["abs_obs", "abs_z", "abs_r2", "nat_obs", "nat_z", "nat_r2"]
    acc = {t: {k: [] for k in keys} for t in range(1, 46)}

    for short in args.proteins.split(","):
        d = ROOT / "runs" / BENCH[short]
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]; L = z_full.shape[2]
        ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
        if len(ii) > args.S:
            sel = rng.choice(len(ii), args.S, replace=False); ii, jj = ii[sel], jj[sel]
        it = torch.as_tensor(ii, device=device); jt = torch.as_tensor(jj, device=device)
        zc = {}
        for t in range(1, 46):
            if t not in zc:
                zc[t] = z_full[t, 0].to(device, torch.float32)
            zc[t + 1] = z_full[t + 1, 0].to(device, torch.float32)
            ell_t = sym_belief(probes[t], zc[t], it, jt, L)
            ell_t1 = sym_belief(probes[t + 1], zc[t + 1], it, jt, L)
            step = ell_t1 - ell_t
            g, ng = grad_natural(ell_t, args.damping)
            a = acc[t]
            obs, r2, m, s = stats(step, -g, args.nperm)
            a["abs_obs"].append(obs); a["abs_r2"].append(r2)
            a["abs_z"].append((obs - m) / (s + 1e-12))
            obs, r2, m, s = stats(step, -ng, args.nperm)
            a["nat_obs"].append(obs); a["nat_r2"].append(r2)
            a["nat_z"].append((obs - m) / (s + 1e-12))
            zc.pop(t, None)
        del z_full, blob
        print(f"  done {short} (L={L})", flush=True)

    rndR2 = 1.0 / 63.0
    print(f"\nrandom per-pair cos^2 baseline (1/Deff, Deff~63) = {rndR2:.4f}", flush=True)
    rows = []
    print(f"\n{'L':>3} | {'abs_obs':>7} {'abs_z':>7} {'abs_R2':>7} | "
          f"{'nat_obs':>7} {'nat_z':>7} {'nat_R2':>7}", flush=True)
    for t in range(1, 46):
        r = {k: float(np.mean(v)) for k, v in acc[t].items()}; r["layer"] = t; rows.append(r)
        print(f"{t:>3} | {r['abs_obs']:>7.4f} {r['abs_z']:>7.1f} {r['abs_r2']:>7.4f} | "
              f"{r['nat_obs']:>7.4f} {r['nat_z']:>7.1f} {r['nat_r2']:>7.4f}", flush=True)

    import csv as _csv
    with open(ROOT / "runs" / "belief_gd_dim.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["layer"] + [k for k in rows[0] if k != "layer"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {ROOT/'runs'/'belief_gd_dim.csv'}", flush=True)

    def blk(lo, hi, k):
        return float(np.mean([r[k] for r in rows if lo <= r["layer"] <= hi]))
    print(f"\n{'block':>12} | {'abs_obs':>7} {'abs_z':>7} {'abs_R2':>7} | "
          f"{'nat_obs':>7} {'nat_z':>7} {'nat_R2':>7}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-45", 32, 45)]:
        print(f"{name:>12} | {blk(lo,hi,'abs_obs'):>7.4f} {blk(lo,hi,'abs_z'):>7.1f} "
              f"{blk(lo,hi,'abs_r2'):>7.4f} | {blk(lo,hi,'nat_obs'):>7.4f} "
              f"{blk(lo,hi,'nat_z'):>7.1f} {blk(lo,hi,'nat_r2'):>7.4f}", flush=True)


if __name__ == "__main__":
    main()
