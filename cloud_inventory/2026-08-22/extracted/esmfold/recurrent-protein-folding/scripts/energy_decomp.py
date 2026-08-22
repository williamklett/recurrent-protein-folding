#!/usr/bin/env python3
"""Decompose the monotone belief-energy decrease into z-motion vs probe-drift.

Belief energy:  E_t = LJ(probe_t(z_t)).   Since A_{t+1}=LJ(probe_{t+1}(z_{t+1}))
the step decomposes EXACTLY:
    E_{t+1}-E_t = [LJ(probe_{t+1}(z_{t+1})) - LJ(probe_{t+1}(z_t))]   (z-motion, fixed probe)
                + [LJ(probe_{t+1}(z_t))     - LJ(probe_t(z_t))]       (probe-drift, fixed z)
No autograd; just forward readouts.
"""
import argparse, csv, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding
from recurrent_folding.esmfold_probes import PairDistogramBottleneck

SIGMA, EPS, SEQ_SEP = 6.8, 1.0, 3
PROBE_DIR = Path("runs/general_belief_replay_benchmark20_cw30")


def centers(device):
    s, e, b = 2.3125, 21.6875, 64
    w = (e - s) / b
    return torch.linspace(s + w / 2, e - w / 2, b, device=device)


def load_probe(layer, device, pair_dim=128):
    p = PROBE_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not p.exists():
        return None
    pr = PairDistogramBottleneck(pair_dim=pair_dim, bins=64, hidden_dim=256).to(device)
    pr.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    pr.eval()
    return pr


def lj_of(probe, z, i_t, j_t, L, cen, keep):
    logits = probe.encode(z.reshape(-1, z.shape[-1]), i_t, j_t, L).reshape(1, L, L, 64)
    logits = 0.5 * (logits + logits.transpose(1, 2))
    ed = (F.softmax(logits, -1) * cen).sum(-1).clamp(min=0.5)
    r = SIGMA / ed
    return float(((4 * EPS * (r ** 12 - r ** 6)) * keep).sum() / L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
    ap.add_argument("--out-csv", default="runs/energy_decomp.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
    model.eval()

    seq = args.sequence
    inputs = tok([seq], return_tensors="pt", add_special_tokens=False).to(device)
    L = len(seq)
    cen = centers(device)
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    i_t, j_t = ii.flatten(), jj.flatten()
    keep = ((jj - ii).abs() > SEQ_SEP).float()

    init = {}
    def hook(_m, a):
        if "z" not in init:
            init["s"] = a[0].detach(); init["z"] = a[1].detach()
    h = model.trunk.blocks[0].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(**inputs, num_recycles=0)
    h.remove()
    s_s, s_z = init["s"], init["z"]
    residx = torch.arange(L, device=device).unsqueeze(0)

    rows = []
    with torch.no_grad():
        for t in range(48):
            pt = load_probe(t, device, pair_dim=s_z.shape[-1])      # probe_t (current)
            pt1 = load_probe(t + 1, device, pair_dim=s_z.shape[-1]) # probe_{t+1}
            z_t = s_z
            s_s, s_z = model.trunk.blocks[t](s_s, s_z, mask=None, residue_index=residx)
            z_t1 = s_z
            if pt is None or pt1 is None:
                continue
            A = lj_of(pt, z_t, i_t, j_t, L, cen, keep)        # belief energy at t
            B = lj_of(pt1, z_t, i_t, j_t, L, cen, keep)        # input z, next probe
            C = lj_of(pt1, z_t1, i_t, j_t, L, cen, keep)       # output z, next probe = E_{t+1}
            rows.append({"layer": t, "E_belief": A, "E_next": C,
                         "dE_total": C - A, "dE_zmotion": C - B, "dE_probedrift": B - A})
            print(f"L{t:02d} | Ebel {A:7.3f} | dTot {C-A:+7.3f} = zmotion {C-B:+7.3f} + drift {B-A:+7.3f}", flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def block(name, lo, hi):
        z = sum(r["dE_zmotion"] for r in rows if lo <= r["layer"] <= hi)
        d = sum(r["dE_probedrift"] for r in rows if lo <= r["layer"] <= hi)
        tot = sum(r["dE_total"] for r in rows if lo <= r["layer"] <= hi)
        up = sum(1 for r in rows if lo <= r["layer"] <= hi and r["dE_zmotion"] > 0)
        n = sum(1 for r in rows if lo <= r["layer"] <= hi)
        print(f"\n[{name} L{lo}-{hi}] total ΔE {tot:+.2f} = z-motion {z:+.2f} + probe-drift {d:+.2f}"
              f"  | z-motion UP in {up}/{n} layers", flush=True)
    block("EARLY", 0, 11); block("MID", 12, 31); block("LATE", 32, 45); block("ALL", 0, 45)


if __name__ == "__main__":
    main()
