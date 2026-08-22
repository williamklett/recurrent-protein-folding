#!/usr/bin/env python3
"""Does the trunk's belief (distogram) trajectory follow gradient descent on the
LJ energy?  Computed entirely in BELIEF space -- one fixed energy E(ell), each layer
read by its own in-distribution probe.  No transport.

State at layer t: symmetrized belief logits  ell_t = encode_t(z_t)  (per pair, 64 bins).
Energy:  E(ell) = sum_pairs LJ( softmax(ell) . centers ).

For step t -> t+1 we test alignment of the belief change with the descent direction,
gradient taken at the SOURCE ell_t (standard GD convention  ell_{t+1} = ell_t - eta*grad):

  ABSOLUTE (Euclidean):  cos( step , -grad_ell E )
  NATURAL  (Fisher)   :  cos( step , -F^+ grad_ell E ),  F = diag(p) - p p^T  (categorical)

Two step definitions (both transport-free):
  content :  encode_t(z_{t+1}) - encode_t(z_t)      fixed dict t, activation moves  (clean)
  total   :  encode_{t+1}(z_{t+1}) - encode_t(z_t)  realized trajectory (content + drift)

Reported per layer: mean per-pair cosine (equal weighting) and global cosine (mag-weighted).
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
BENCH = {"ubq": "benchmark_ubq", "mbn": "benchmark_mbn", "1AO6": "benchmark_1AO6",
         "T1170": "benchmark_T1170", "7ACN": "benchmark_7ACN"}


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
    ap.add_argument("--out-csv", default=str(ROOT / "runs" / "belief_gd.csv"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)
    probes = {t: load_probe(t, device) for t in range(1, 47)}

    def sym_belief(pr, zmap, it, jt, L):
        lij = pr.encode(zmap[it, jt], it, jt, L)
        lji = pr.encode(zmap[jt, it], jt, it, L)
        return 0.5 * (lij + lji)

    def grad_natural(ell, damping):
        L0 = ell.detach().clone().requires_grad_(True)
        p = F.softmax(L0, -1)
        ed = (p * cen).sum(-1).clamp(min=0.5)
        rr = SIGMA / ed
        E = (4 * EPS * (rr ** 12 - rr ** 6)).sum()
        g = torch.autograd.grad(E, L0)[0]                       # (N,64)
        with torch.no_grad():
            pp = F.softmax(L0, -1)
            Fm = torch.diag_embed(pp) - pp[..., :, None] * pp[..., None, :]   # (N,64,64)
            lam = damping * (pp - pp ** 2).mean(-1, keepdim=True)             # (N,1)
            Fd = Fm + lam[..., None] * torch.eye(64, device=device)
            x = torch.linalg.solve(Fd, g.unsqueeze(-1)).squeeze(-1)           # F^+ g
        return g.detach(), x

    def cosines(step, neg_dir):
        pp = F.cosine_similarity(step, neg_dir, dim=-1).mean().item()         # per-pair mean
        gg = (step * neg_dir).sum().item() / (step.norm().item() * neg_dir.norm().item() + 1e-12)
        return pp, gg

    acc = {t: {k: [] for k in ["c_abs_pp", "c_abs_g", "c_nat_pp", "c_nat_g",
                                "t_abs_pp", "t_abs_g", "t_nat_pp", "t_nat_g", "gnat_align"]}
           for t in range(1, 46)}

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
            pt, pt1 = probes[t], probes[t + 1]
            if t not in zc:
                zc[t] = z_full[t, 0].to(device, torch.float32)
            zc[t + 1] = z_full[t + 1, 0].to(device, torch.float32)
            ell_t = sym_belief(pt, zc[t], it, jt, L)
            ell_t_t1 = sym_belief(pt, zc[t + 1], it, jt, L)
            ell_t1 = sym_belief(pt1, zc[t + 1], it, jt, L)
            content = ell_t_t1 - ell_t
            total = ell_t1 - ell_t
            g, ng = grad_natural(ell_t, args.damping)
            a = acc[t]
            pp, gg = cosines(content, -g);  a["c_abs_pp"].append(pp); a["c_abs_g"].append(gg)
            pp, gg = cosines(content, -ng); a["c_nat_pp"].append(pp); a["c_nat_g"].append(gg)
            pp, gg = cosines(total, -g);    a["t_abs_pp"].append(pp); a["t_abs_g"].append(gg)
            pp, gg = cosines(total, -ng);   a["t_nat_pp"].append(pp); a["t_nat_g"].append(gg)
            a["gnat_align"].append(F.cosine_similarity(g, ng, dim=-1).mean().item())
            zc.pop(t, None)
        del z_full, blob
        print(f"  done {short} (L={L})", flush=True)

    floor = 1.0 / (64 * min(args.S, L * (L - 1) // 2)) ** 0.5
    print(f"\nnoise floor (per-pair mean) ~ {floor:.4f}", flush=True)
    print(f"\n{'L':>3} | {'CONTENT abs':>11} {'CONTENT nat':>11} | "
          f"{'TOTAL abs':>10} {'TOTAL nat':>10} | {'g.nat':>6}", flush=True)
    rows = []
    for t in range(1, 46):
        r = {k: float(np.mean(v)) for k, v in acc[t].items()}
        r["layer"] = t
        rows.append(r)
        print(f"{t:>3} | {r['c_abs_pp']:>11.4f} {r['c_nat_pp']:>11.4f} | "
              f"{r['t_abs_pp']:>10.4f} {r['t_nat_pp']:>10.4f} | {r['gnat_align']:>6.3f}", flush=True)

    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer"] + [k for k in rows[0] if k != "layer"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out_csv}", flush=True)

    def blk(lo, hi, k):
        return float(np.mean([r[k] for r in rows if lo <= r["layer"] <= hi]))
    print(f"\n{'block':>10} | {'C_abs':>7} {'C_nat':>7} | {'T_abs':>7} {'T_nat':>7}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-45", 32, 45)]:
        print(f"{name:>10} | {blk(lo,hi,'c_abs_pp'):>7.4f} {blk(lo,hi,'c_nat_pp'):>7.4f} | "
              f"{blk(lo,hi,'t_abs_pp'):>7.4f} {blk(lo,hi,'t_nat_pp'):>7.4f}", flush=True)


if __name__ == "__main__":
    main()
