#!/usr/bin/env python3
"""Can we find a CODE change Dz that makes a STALE dictionary read CORRECTLY?

For activation z_target (= z_full[t]), the correct dictionary is encode_t; the stale
one is encode_{t-k}.  We solve, by backprop through the ENCODER (robust direction):

    Dz* = argmin_Dz || encode_{t-k}(z_t + Dz) - encode_t(z_t) ||^2 ,  init Dz=0

Implicit minimum-norm: starting at 0, GD keeps Dz in the encoder Jacobian row space
(orthogonal to its 64-dim null space).  We report:
  bel_before : EDrms(encode_{t-k}(z_t),        encode_t(z_t))   stale error
  bel_after  : EDrms(encode_{t-k}(z_t + Dz*),  encode_t(z_t))   residual after correction
  dz_rel     : ||Dz*|| / ||z_t||                                size of needed code change
  cos_step   : cos(Dz*, z_t - z_{t-k})    alignment of correction w/ actual inter-layer motion
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


def correct(stale, z, it, jt, L, target_logits, steps, lr):
    dz = torch.zeros_like(z, requires_grad=True)
    opt = torch.optim.Adam([dz], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = stale.encode(z + dz, it, jt, L)
        loss = ((out - target_logits) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    return dz.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=800)
    ap.add_argument("--n-prot", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--ks", default="1,5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)
    ks = [int(x) for x in args.ks.split(",")]

    dirs = [d for d in sorted((ROOT / "runs").glob("synthetic_design_*"))
            if (d / "esmfold_activations.pt").exists()]
    dirs = [dirs[i] for i in rng.permutation(len(dirs))[: args.n_prot]]
    print("proteins:", [d.name for d in dirs], "  ks:", ks, flush=True)
    probes = {t: load_probe(t, device) for t in range(1, 47)}

    def ED(lg):
        return (F.softmax(lg, -1) * cen).sum(-1)

    def edrms(a, b):
        return float(((ED(a) - ED(b)) ** 2).mean() ** 0.5)

    agg = {k: {m: [] for m in ["before", "after", "dz_rel", "cos_step"]} for k in ks}

    for d in dirs:
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]; L = z_full.shape[2]
        ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
        if len(ii) > args.S:
            sel = rng.choice(len(ii), args.S, replace=False); ii, jj = ii[sel], jj[sel]
        it = torch.as_tensor(ii, device=device); jt = torch.as_tensor(jj, device=device)
        for k in ks:
            for t in range(1 + k, 46):
                stale = probes[t - k]
                z_t = z_full[t, 0].to(device, torch.float32)[it, jt]
                z_tk = z_full[t - k, 0].to(device, torch.float32)[it, jt]
                tgt = probes[t].encode(z_t, it, jt, L)        # correct belief
                before = edrms(stale.encode(z_t, it, jt, L), tgt)
                dz = correct(stale, z_t, it, jt, L, tgt, args.steps, args.lr)
                after = edrms(stale.encode(z_t + dz, it, jt, L), tgt)
                dz_rel = float((dz.norm(dim=-1) / z_t.norm(dim=-1).clamp_min(1e-6)).mean())
                step = z_t - z_tk
                cos = F.cosine_similarity(dz, step, dim=-1).mean().item()
                agg[k]["before"].append(before); agg[k]["after"].append(after)
                agg[k]["dz_rel"].append(dz_rel); agg[k]["cos_step"].append(cos)
        del z_full, blob
        print(f"  done {d.name}", flush=True)

    print(f"\n{'k(stale)':>8} | {'bel_before(A)':>13} {'bel_after(A)':>12} "
          f"{'dz_rel':>8} {'cos_step':>8}", flush=True)
    for k in ks:
        a = agg[k]
        print(f"{k:>8} | {np.mean(a['before']):>13.4f} {np.mean(a['after']):>12.4f} "
              f"{np.mean(a['dz_rel']):>8.4f} {np.mean(a['cos_step']):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
