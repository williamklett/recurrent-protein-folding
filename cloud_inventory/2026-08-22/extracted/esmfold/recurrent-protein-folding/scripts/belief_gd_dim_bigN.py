#!/usr/bin/env python3
"""Larger-N dimension-controlled belief-GD / cosine test.

Same measure as belief_gd_dim.py but auto-discovers ALL runs/benchmark_* proteins
that have esmfold_activations.pt (real benchmark set, N up to 20), so the cosine
alignment is reported on a much larger sample than the original N=3.

ell_t = encode_t(z_t);  step = ell_{t+1} - ell_t;  g = grad_ell E(ell_t).
Per pair the LJ gradient is ~1-D, so the descent target is ~1-D and ~62 dims are
energy-neutral.  We report the realized alignment vs a dimension- and magnitude-
matched permutation null.

  *_obs : observed global (magnitude-weighted) cosine
  *_z   : (obs - null_mean)/null_std   (dimension-free effect size)
  *_r2  : mean over pairs of cos^2(step, grad)   (vs 1/Deff ~ 1/63 = 0.0159 random)
abs_ = Euclidean gradient; nat_ = Fisher-natural gradient.
Also reports per-protein so we can see the spread across the larger sample.
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


def discover_proteins():
    out = []
    for p in sorted((ROOT / "runs").glob("benchmark_*/esmfold_activations.pt")):
        out.append((p.parent.name.replace("benchmark_", ""), p.parent))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", default="all", help="'all' or comma list of short names")
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

    allp = discover_proteins()
    if args.proteins != "all":
        keep = set(args.proteins.split(","))
        allp = [(n, d) for n, d in allp if n in keep]
    print(f"Running on N={len(allp)} proteins: {[n for n,_ in allp]}", flush=True)

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
        cpp = F.cosine_similarity(step, neg_dir, dim=-1)
        r2 = (cpp ** 2).mean().item()
        nums = torch.empty(nperm, device=device)
        denom = sd * nd + 1e-12
        for q in range(nperm):
            perm = torch.randperm(N, device=device, generator=g_cpu)
            nums[q] = (step * neg_dir[perm]).sum() / denom
        return obs, r2, nums.mean().item(), nums.std().item()

    g_cpu.manual_seed(args.seed)
    keys = ["abs_obs", "abs_z", "abs_r2", "nat_obs", "nat_z", "nat_r2", "dE", "frac_dn"]
    acc = {t: {k: [] for k in keys} for t in range(1, 46)}
    per_prot = {}   # name -> dict of block-averaged metrics

    for short, d in allp:
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]; L = z_full.shape[2]
        ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
        if len(ii) > args.S:
            sel = rng.choice(len(ii), args.S, replace=False); ii, jj = ii[sel], jj[sel]
        it = torch.as_tensor(ii, device=device); jt = torch.as_tensor(jj, device=device)
        ploc = {k: [] for k in keys}
        zc = {}
        for t in range(1, 46):
            if t not in zc:
                zc[t] = z_full[t, 0].to(device, torch.float32)
            zc[t + 1] = z_full[t + 1, 0].to(device, torch.float32)
            ell_t = sym_belief(probes[t], zc[t], it, jt, L)
            ell_t1 = sym_belief(probes[t + 1], zc[t + 1], it, jt, L)
            step = ell_t1 - ell_t
            g, ng = grad_natural(ell_t, args.damping)
            p0 = F.softmax(ell_t, -1); ed0 = (p0 * cen).sum(-1).clamp(min=0.5); r0 = SIGMA / ed0
            e0 = 4 * EPS * (r0 ** 12 - r0 ** 6)
            p1 = F.softmax(ell_t1, -1); ed1 = (p1 * cen).sum(-1).clamp(min=0.5); r1 = SIGMA / ed1
            e1 = 4 * EPS * (r1 ** 12 - r1 ** 6)
            a = acc[t]
            obs, r2, m, s = stats(step, -g, args.nperm)
            a["abs_obs"].append(obs); a["abs_r2"].append(r2); a["abs_z"].append((obs - m) / (s + 1e-12))
            ploc["abs_obs"].append(obs); ploc["abs_r2"].append(r2); ploc["abs_z"].append((obs - m) / (s + 1e-12))
            obs, r2, m, s = stats(step, -ng, args.nperm)
            a["nat_obs"].append(obs); a["nat_r2"].append(r2); a["nat_z"].append((obs - m) / (s + 1e-12))
            ploc["nat_obs"].append(obs); ploc["nat_r2"].append(r2); ploc["nat_z"].append((obs - m) / (s + 1e-12))
            dE = (e1 - e0).sum().item(); fdn = (e1 < e0).float().mean().item()
            a["dE"].append(dE); a["frac_dn"].append(fdn)
            ploc["dE"].append(dE); ploc["frac_dn"].append(fdn)
            zc.pop(t, None)
        per_prot[short] = {k: float(np.mean(v)) for k, v in ploc.items()}
        del z_full, blob
        torch.cuda.empty_cache()
        print(f"  done {short} (L={L})  abs_obs={per_prot[short]['abs_obs']:+.4f} "
              f"nat_obs={per_prot[short]['nat_obs']:+.4f} dE={per_prot[short]['dE']:+.2f}", flush=True)

    # per-layer aggregate
    rndR2 = 1.0 / 63.0
    print(f"\nrandom per-pair cos^2 baseline (1/Deff~63) = {rndR2:.4f}", flush=True)
    rows = []
    for t in range(1, 46):
        r = {k: float(np.mean(v)) for k, v in acc[t].items()}; r["layer"] = t; rows.append(r)
    import csv as _csv
    with open(ROOT / "runs" / "belief_gd_dim_bigN.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["layer"] + [k for k in rows[0] if k != "layer"])
        w.writeheader(); w.writerows(rows)
    with open(ROOT / "runs" / "belief_gd_dim_bigN_perprotein.csv", "w", newline="") as f:
        flds = ["protein"] + keys
        w = _csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for name, dd in per_prot.items():
            w.writerow({"protein": name, **dd})
    print(f"\nWrote belief_gd_dim_bigN.csv and belief_gd_dim_bigN_perprotein.csv", flush=True)

    def blk(lo, hi, k):
        return float(np.mean([r[k] for r in rows if lo <= r["layer"] <= hi]))
    print(f"\n{'block':>12} | {'abs_obs':>7} {'abs_z':>7} {'abs_R2':>7} | "
          f"{'nat_obs':>7} {'nat_z':>7} {'nat_R2':>7} | {'dE':>7} {'frac_dn':>7}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-45", 32, 45)]:
        print(f"{name:>12} | {blk(lo,hi,'abs_obs'):>7.4f} {blk(lo,hi,'abs_z'):>7.1f} "
              f"{blk(lo,hi,'abs_r2'):>7.4f} | {blk(lo,hi,'nat_obs'):>7.4f} "
              f"{blk(lo,hi,'nat_z'):>7.1f} {blk(lo,hi,'nat_r2'):>7.4f} | "
              f"{blk(lo,hi,'dE'):>7.2f} {blk(lo,hi,'frac_dn'):>7.3f}", flush=True)

    # overall cosine summary across all proteins/layers
    ao = float(np.mean([r["abs_obs"] for r in rows]))
    no = float(np.mean([r["nat_obs"] for r in rows]))
    ar = float(np.mean([r["abs_r2"] for r in rows]))
    nr = float(np.mean([r["nat_r2"] for r in rows]))
    print(f"\nOVERALL (N={len(allp)} proteins): abs_obs={ao:+.4f} nat_obs={no:+.4f} "
          f"abs_R2={ar:.4f} nat_R2={nr:.4f}  (random R2={rndR2:.4f})", flush=True)
    print("DONE_BIGN", flush=True)


if __name__ == "__main__":
    main()
