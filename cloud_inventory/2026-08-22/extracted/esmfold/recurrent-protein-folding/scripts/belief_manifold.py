#!/usr/bin/env python3
"""Does the trunk RESPECT the energy manifold (vs do GD)?

In-distribution readouts: ell_t = encode_t(z_t); step = ell_{t+1}-ell_t; g=grad E(ell_t).
Decompose per pair:  step = step_par (along g) + step_perp (tangent to level set).

Tests (all finite-step, energy E=sum_pairs LJ(softmax(ell).centers)):
  dE_act      : E(ell_t+step)      - E(ell_t)     actual fold (should be <0)
  dE_iso      : E(ell_t+r_iso)     - E(ell_t)     r_iso = isotropic, ||r_iso_pair||=||step_pair||
  dE_perp_act : E(ell_t+step_perp) - E(ell_t)     trunk's tangent-only motion
  dE_perp_rnd : E(ell_t+r_perp)    - E(ell_t)     random vec projected off g, matched ||step_perp||

Manifold-respecting signature:
  (a) dE_iso >> 0      generic same-size moves CLIMB (the manifold is narrow)
  (b) |dE_perp_act| << |dE_perp_rnd|   trunk's tangent motion is energy-preserving
      (it curves WITH the level set; a random tangent move of equal size does not)
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
    ap.add_argument("--nrep", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    probes = {t: load_probe(t, device) for t in range(1, 47)}

    def sym_belief(pr, zmap, it, jt, L):
        lij = pr.encode(zmap[it, jt], it, jt, L)
        lji = pr.encode(zmap[jt, it], jt, it, L)
        return 0.5 * (lij + lji)

    def Epp(ell):
        p = F.softmax(ell, -1); ed = (p * cen).sum(-1).clamp(min=0.5)
        rr = SIGMA / ed
        return 4 * EPS * (rr ** 12 - rr ** 6)

    def grad(ell):
        L0 = ell.detach().clone().requires_grad_(True)
        E = Epp(L0).sum()
        return torch.autograd.grad(E, L0)[0].detach()

    def pnorm(x):
        return x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    keys = ["dE_act", "dE_iso", "dE_perp_act", "dE_perp_rnd"]
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
            g = grad(ell_t); gu = g / pnorm(g)
            e0 = Epp(ell_t)
            sn = pnorm(step)                                   # ||step_pair||
            step_perp = step - (step * gu).sum(-1, keepdim=True) * gu
            spn = pnorm(step_perp)
            a = acc[t]
            a["dE_act"].append((Epp(ell_t + step) - e0).sum().item())
            a["dE_perp_act"].append((Epp(ell_t + step_perp) - e0).sum().item())
            iso_l, perp_l = [], []
            for _ in range(args.nrep):
                r = torch.randn(step.shape, generator=gen, device=device)
                r = r / pnorm(r) * sn                          # isotropic, matched norm
                iso_l.append((Epp(ell_t + r) - e0).sum().item())
                rp = torch.randn(step.shape, generator=gen, device=device)
                rp = rp - (rp * gu).sum(-1, keepdim=True) * gu  # project off g
                rp = rp / pnorm(rp) * spn                       # matched to ||step_perp||
                perp_l.append((Epp(ell_t + rp) - e0).sum().item())
            a["dE_iso"].append(float(np.mean(iso_l)))
            a["dE_perp_rnd"].append(float(np.mean(perp_l)))
            zc.pop(t, None)
        del z_full, blob
        print(f"  done {short} (L={L})", flush=True)

    rows = []
    print(f"\n{'L':>3} | {'dE_act':>9} {'dE_iso':>9} | {'perp_act':>9} {'perp_rnd':>9} "
          f"{'ratio':>6}", flush=True)
    for t in range(1, 46):
        r = {k: float(np.mean(v)) for k, v in acc[t].items()}; r["layer"] = t; rows.append(r)
        ratio = abs(r["dE_perp_act"]) / (abs(r["dE_perp_rnd"]) + 1e-9)
        print(f"{t:>3} | {r['dE_act']:>9.2f} {r['dE_iso']:>9.1f} | "
              f"{r['dE_perp_act']:>9.2f} {r['dE_perp_rnd']:>9.1f} {ratio:>6.3f}", flush=True)

    def blk(lo, hi, k):
        return float(np.mean([r[k] for r in rows if lo <= r["layer"] <= hi]))
    print(f"\n{'block':>12} | {'dE_act':>9} {'dE_iso':>9} | {'perp_act':>9} {'perp_rnd':>9} "
          f"{'ratio':>6}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-45", 32, 45)]:
        pa, pr = blk(lo, hi, "dE_perp_act"), blk(lo, hi, "dE_perp_rnd")
        print(f"{name:>12} | {blk(lo,hi,'dE_act'):>9.2f} {blk(lo,hi,'dE_iso'):>9.1f} | "
              f"{pa:>9.2f} {pr:>9.1f} {abs(pa)/(abs(pr)+1e-9):>6.3f}", flush=True)


if __name__ == "__main__":
    main()
