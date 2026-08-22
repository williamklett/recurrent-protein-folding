#!/usr/bin/env python3
"""CLEAN belief-GD test.  No fixed/OOD dictionary.

Every layer read IN-DISTRIBUTION by its own probe:
    ell_t = encode_t(z_t)                 (symmetrized, 64 bins/pair)
Realized step:
    step  = ell_{t+1} - ell_t
Energy of expected shape:
    E(ell) = sum_pairs LJ( softmax(ell) . centers ),  grad taken at source ell_t
Directions:
    absolute = -grad_ell E
    natural  = -F^+ grad_ell E,  F = diag(p) - p p^T

For each step report:
    dE         : E(ell_{t+1}) - E(ell_t)            (should be < 0; the fold)
    abs_pp/nat_pp : per-pair MEAN cosine (democratic)
    abs_g /nat_g  : GLOBAL magnitude-weighted cosine -- its sign tracks dE
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

    def energy_per_pair(ell):
        p = F.softmax(ell, -1)
        ed = (p * cen).sum(-1).clamp(min=0.5)
        rr = SIGMA / ed
        return 4 * EPS * (rr ** 12 - rr ** 6)            # (N,)

    def grad_natural(ell, damping):
        L0 = ell.detach().clone().requires_grad_(True)
        E = energy_per_pair(L0).sum()
        g = torch.autograd.grad(E, L0)[0]
        with torch.no_grad():
            pp = F.softmax(L0, -1)
            Fm = torch.diag_embed(pp) - pp[..., :, None] * pp[..., None, :]
            lam = damping * (pp - pp ** 2).mean(-1, keepdim=True)
            Fd = Fm + lam[..., None] * torch.eye(64, device=device)
            x = torch.linalg.solve(Fd, g.unsqueeze(-1)).squeeze(-1)
        return g.detach(), x

    def cos_pp_g(step, neg_dir):
        pp = F.cosine_similarity(step, neg_dir, dim=-1).mean().item()
        gg = (step * neg_dir).sum().item() / (step.norm().item() * neg_dir.norm().item() + 1e-12)
        return pp, gg

    keys = ["dE", "abs_pp", "abs_g", "nat_pp", "nat_g", "frac_dn"]
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
            e0 = energy_per_pair(ell_t); e1 = energy_per_pair(ell_t1)
            a = acc[t]
            a["dE"].append((e1 - e0).sum().item())
            a["frac_dn"].append((e1 < e0).float().mean().item())
            pp, gg = cos_pp_g(step, -g);  a["abs_pp"].append(pp); a["abs_g"].append(gg)
            pp, gg = cos_pp_g(step, -ng); a["nat_pp"].append(pp); a["nat_g"].append(gg)
            zc.pop(t, None)
        del z_full, blob
        print(f"  done {short} (L={L})", flush=True)

    floor = 1.0 / (64 * min(args.S, L * (L - 1) // 2)) ** 0.5
    print(f"\nnoise floor (per-pair mean) ~ {floor:.4f}", flush=True)
    print(f"\n{'L':>3} | {'dE':>9} {'frac_dn':>7} | {'abs_pp':>7} {'abs_g':>7} | "
          f"{'nat_pp':>7} {'nat_g':>7}", flush=True)
    rows = []
    for t in range(1, 46):
        r = {k: float(np.mean(v)) for k, v in acc[t].items()}
        r["layer"] = t; rows.append(r)
        print(f"{t:>3} | {r['dE']:>9.3f} {r['frac_dn']:>7.3f} | "
              f"{r['abs_pp']:>7.4f} {r['abs_g']:>7.4f} | "
              f"{r['nat_pp']:>7.4f} {r['nat_g']:>7.4f}", flush=True)

    import csv as _csv
    with open(ROOT / "runs" / "belief_gd_clean.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["layer"] + [k for k in rows[0] if k != "layer"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {ROOT/'runs'/'belief_gd_clean.csv'}", flush=True)

    def blk(lo, hi, k):
        return float(np.mean([r[k] for r in rows if lo <= r["layer"] <= hi]))
    print(f"\n{'block':>12} | {'dE_sum':>8} {'frac_dn':>7} | {'abs_pp':>7} {'abs_g':>7} | "
          f"{'nat_pp':>7} {'nat_g':>7}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-45", 32, 45)]:
        print(f"{name:>12} | {blk(lo,hi,'dE'):>8.3f} {blk(lo,hi,'frac_dn'):>7.3f} | "
              f"{blk(lo,hi,'abs_pp'):>7.4f} {blk(lo,hi,'abs_g'):>7.4f} | "
              f"{blk(lo,hi,'nat_pp'):>7.4f} {blk(lo,hi,'nat_g'):>7.4f}", flush=True)


if __name__ == "__main__":
    main()
