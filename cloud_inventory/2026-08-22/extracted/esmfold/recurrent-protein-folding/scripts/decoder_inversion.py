#!/usr/bin/env python3
"""Decoder-inversion readout: read an activation's belief through layer-t's lens by
solving  ell* = argmin_ell || decode_t(ell) - z ||   via backprop through decode_t.

This is over-determined (64 logits -> 128-dim activation target), hence well-posed,
unlike inverting the many-to-one encoder.  We test whether it lets the FIXED layer-t
probe faithfully read a NEXT-layer activation z_{t+1}.

Per held-out protein, per layer t (probe_t reads z_full[t]; z_{t+1}=z_full[t+1]):

  SELF (in-distribution, z_t):
    recon_ff   = ||decode_t(encode_t(z_t)) - z_t||_rel        (feedforward floor ~0.44)
    recon_inv  = ||decode_t(ell*_self)     - z_t||_rel        (best decoder fit)
    bel_inv_enc= EDrms(softmax ell*_self, softmax encode_t(z_t))   (does inv agree w/ encoder?)

  CROSS (read z_{t+1} through t's decoder):
    recon_cross= ||decode_t(ell*_cross) - z_{t+1}||_rel
    inv_vs_true= EDrms(softmax ell*_cross, true belief encode_{t+1}(z_{t+1}))  <-- KEY
    ff_vs_true = EDrms(softmax encode_t(z_{t+1}), true belief)                 (stale baseline)

KEY question: is inv_vs_true (decoder-inversion transport) << ff_vs_true (stale)?
If yes, probe_t can faithfully decode z_{t+1} after decoder-inversion.
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


def load_probe(layer, device, pair_dim=128):
    p = PROBE_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not p.exists():
        return None
    pr = PairDistogramBottleneck(pair_dim=pair_dim, bins=64, hidden_dim=256).to(device)
    pr.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    pr.eval()
    for q in pr.parameters():
        q.requires_grad = False
    return pr


def invert_decoder(probe, z_tgt, it, jt, L, init, steps, lr):
    """ell* = argmin ||decode(ell) - z_tgt||^2, backprop through decoder."""
    ell = init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([ell], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        zrec = probe.decode(ell, it, jt, L)
        loss = ((zrec - z_tgt) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
    return ell.detach()


def rel_recon(zrec, z):
    return float((((zrec - z) ** 2).sum(-1) / (z ** 2).sum(-1).clamp_min(1e-6)).mean() ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=800)
    ap.add_argument("--n-prot", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)

    dirs = [d for d in sorted((ROOT / "runs").glob("synthetic_design_*"))
            if (d / "esmfold_activations.pt").exists()]
    dirs = [dirs[i] for i in rng.permutation(len(dirs))[: args.n_prot]]
    print("proteins:", [d.name for d in dirs], flush=True)

    probes = {t: load_probe(t, device) for t in range(1, 47)}

    def ED(logits):
        return (F.softmax(logits, -1) * cen).sum(-1)

    def edrms(a, b):
        return float(((ED(a) - ED(b)) ** 2).mean() ** 0.5)

    keys = ["recon_ff", "recon_inv", "bel_inv_enc", "recon_cross", "inv_vs_true", "ff_vs_true"]
    agg = {t: {k: [] for k in keys} for t in range(1, 46)}

    for d in dirs:
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]; L = z_full.shape[2]
        ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
        if len(ii) > args.S:
            sel = rng.choice(len(ii), args.S, replace=False); ii, jj = ii[sel], jj[sel]
        it = torch.as_tensor(ii, device=device); jt = torch.as_tensor(jj, device=device)
        for t in range(1, 46):
            pt, pt1 = probes[t], probes[t + 1]
            z_t = z_full[t, 0].to(device, torch.float32)[it, jt]
            z_t1 = z_full[t + 1, 0].to(device, torch.float32)[it, jt]
            ell_t = pt.encode(z_t, it, jt, L)            # feedforward t-belief of z_t
            ell_t1_true = pt1.encode(z_t1, it, jt, L)     # true belief of z_{t+1}
            ell_t_on_t1 = pt.encode(z_t1, it, jt, L)      # stale feedforward
            # self feedforward recon floor
            agg[t]["recon_ff"].append(rel_recon(pt.decode(ell_t, it, jt, L), z_t))
            # self decoder-inversion
            ell_self = invert_decoder(pt, z_t, it, jt, L, ell_t, args.steps, args.lr)
            agg[t]["recon_inv"].append(rel_recon(pt.decode(ell_self, it, jt, L), z_t))
            agg[t]["bel_inv_enc"].append(edrms(ell_self, ell_t))
            # cross: read z_{t+1} through t's decoder (init from stale feedforward)
            ell_cross = invert_decoder(pt, z_t1, it, jt, L, ell_t_on_t1, args.steps, args.lr)
            agg[t]["recon_cross"].append(rel_recon(pt.decode(ell_cross, it, jt, L), z_t1))
            agg[t]["inv_vs_true"].append(edrms(ell_cross, ell_t1_true))
            agg[t]["ff_vs_true"].append(edrms(ell_t_on_t1, ell_t1_true))
        del z_full, blob
        print(f"  done {d.name}", flush=True)

    print(f"\n{'L':>3} | {'recon_ff':>8} {'recon_inv':>9} {'bel_inv_enc':>11} | "
          f"{'recon_cr':>8} {'inv_vs_true':>11} {'ff_vs_true':>10}", flush=True)
    rows = []
    for t in range(1, 46):
        r = {k: float(np.mean(agg[t][k])) for k in keys}
        rows.append((t, r))
        print(f"{t:>3} | {r['recon_ff']:>8.4f} {r['recon_inv']:>9.4f} {r['bel_inv_enc']:>11.4f} | "
              f"{r['recon_cross']:>8.4f} {r['inv_vs_true']:>11.4f} {r['ff_vs_true']:>10.4f}", flush=True)

    def blk(lo, hi, k):
        return float(np.mean([r[k] for t, r in rows if lo <= t <= hi]))
    print(f"\n{'block':>10} | {'recon_ff':>8} {'recon_inv':>9} {'bel_inv_enc':>11} | "
          f"{'recon_cr':>8} {'inv_vs_true':>11} {'ff_vs_true':>10}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-45", 32, 45)]:
        print(f"{name:>10} | {blk(lo,hi,'recon_ff'):>8.4f} {blk(lo,hi,'recon_inv'):>9.4f} "
              f"{blk(lo,hi,'bel_inv_enc'):>11.4f} | {blk(lo,hi,'recon_cross'):>8.4f} "
              f"{blk(lo,hi,'inv_vs_true'):>11.4f} {blk(lo,hi,'ff_vs_true'):>10.4f}", flush=True)


if __name__ == "__main__":
    main()
