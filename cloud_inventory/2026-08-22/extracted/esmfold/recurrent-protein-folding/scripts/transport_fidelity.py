#!/usr/bin/env python3
"""Validate the analytic cross-layer semantic transform  Phi_{t->s} = decode_s o encode_t
built from the general bottleneck probes (encode = interpretation, decode = inverse).

We measure three things on HELD-OUT proteins, per layer t (1..46):

  (1) SELF round-trip  decode_t(encode_t(z_t)) vs z_t
        -> how good the probe autoencoder is in-distribution (rel L2 on z, and the
           induced belief drift).  Bounds the fidelity of using learned decoders.

  (2) LEARNED transport to reference s (=46, the native-anchored frame):
        zeta = decode_s(encode_t(z_t));  belief_back = encode_s(zeta)
        report expected-distance RMS(A) and KL between belief_back and the faithful
        belief encode_t(z_t).  This is whether Phi_{t->s} lands layer t in s's frame.

  (3) ANALYTIC transport (native pseudo-inverse): zeta' = W^+(ell - b),
        native(zeta') = W zeta' + b.  Should reproduce ell EXACTLY (W full row-rank).
        Reported as a control (expect ~0).

Belief = symmetrized softmax logits; expected distance over bins, pairs |i-j|>SEQ_SEP.
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
REF = 46


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


def load_native_head(device):
    from transformers import EsmForProteinFolding
    m = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    head = m.distogram_head if hasattr(m, "distogram_head") else \
        next(mod for n, mod in m.named_modules() if n.endswith("distogram_head"))
    W = head.weight.detach().to(device, torch.float32).clone()
    b = head.bias.detach().to(device, torch.float32).clone()
    del m
    import gc; gc.collect(); torch.cuda.empty_cache()
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=2000)
    ap.add_argument("--n-prot", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)
    rng = np.random.default_rng(args.seed)

    # held-out synthetic proteins (never seen by the probes)
    dirs = [d for d in sorted((ROOT / "runs").glob("synthetic_design_*"))
            if (d / "esmfold_activations.pt").exists()]
    dirs = [dirs[i] for i in rng.permutation(len(dirs))[: args.n_prot]]
    print("proteins:", [d.name for d in dirs], flush=True)

    W, b = load_native_head(device)
    Wpinv = torch.linalg.pinv(W)
    probes = {t: load_probe(t, device) for t in range(1, 47)}
    ref = probes[REF]

    def ED(logits):
        return (F.softmax(logits, -1) * cen).sum(-1)

    def kl(p_logits, q_logits):
        p = F.softmax(p_logits, -1); lp = F.log_softmax(p_logits, -1); lq = F.log_softmax(q_logits, -1)
        return float((p * (lp - lq)).sum(-1).mean())

    agg = {t: {"recon_z": [], "recon_bel": [], "learn_ed": [], "learn_kl": [], "anal_ed": []}
           for t in range(1, 47)}

    for d in dirs:
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]; L = z_full.shape[2]
        ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
        if len(ii) > args.S:
            sel = rng.choice(len(ii), args.S, replace=False); ii, jj = ii[sel], jj[sel]
        it = torch.as_tensor(ii, device=device); jt = torch.as_tensor(jj, device=device)
        for t in range(1, 47):
            pr = probes[t]
            zt = z_full[t, 0].to(device, torch.float32)
            z = zt[it, jt]                                  # (N,128)
            ell = pr.encode(z, it, jt, L)                   # faithful belief at t
            # (1) self round-trip
            zrec = pr.decode(ell, it, jt, L)
            agg[t]["recon_z"].append(float((((zrec - z) ** 2).sum(-1) / (z ** 2).sum(-1).clamp_min(1e-6)).mean() ** 0.5))
            ell_self = pr.encode(zrec, it, jt, L)
            agg[t]["recon_bel"].append(float(((ED(ell_self) - ED(ell)) ** 2).mean() ** 0.5))
            # (2) learned transport to reference frame s=REF, read back with ref encoder
            zeta = ref.decode(ell, it, jt, L)
            ell_back = ref.encode(zeta, it, jt, L)
            agg[t]["learn_ed"].append(float(((ED(ell_back) - ED(ell)) ** 2).mean() ** 0.5))
            agg[t]["learn_kl"].append(kl(ell, ell_back))
            # (3) analytic native pseudo-inverse control
            zeta2 = (ell - b) @ Wpinv.t()
            ell_anal = zeta2 @ W.t() + b
            agg[t]["anal_ed"].append(float(((ED(ell_anal) - ED(ell)) ** 2).mean() ** 0.5))
            del zt
        del z_full, blob
        print(f"  done {d.name}", flush=True)

    print(f"\n{'L':>3} | {'recon_z(rel)':>12} {'recon_bel(A)':>12} | "
          f"{'learn_ed(A)':>11} {'learn_kl':>9} | {'anal_ed(A)':>10}", flush=True)
    rows = []
    for t in range(1, 47):
        a = agg[t]
        r = {k: float(np.mean(v)) for k, v in a.items()}
        rows.append((t, r))
        print(f"{t:>3} | {r['recon_z']:>12.4f} {r['recon_bel']:>12.4f} | "
              f"{r['learn_ed']:>11.4f} {r['learn_kl']:>9.5f} | {r['anal_ed']:>10.6f}", flush=True)

    def blk(lo, hi, key):
        return float(np.mean([r[key] for t, r in rows if lo <= t <= hi]))
    print(f"\n{'block':>10} | {'recon_z':>8} {'recon_bel':>10} {'learn_ed':>9} {'learn_kl':>9} {'anal_ed':>9}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-46", 32, 46)]:
        print(f"{name:>10} | {blk(lo,hi,'recon_z'):>8.4f} {blk(lo,hi,'recon_bel'):>10.4f} "
              f"{blk(lo,hi,'learn_ed'):>9.4f} {blk(lo,hi,'learn_kl'):>9.5f} {blk(lo,hi,'anal_ed'):>9.6f}", flush=True)


if __name__ == "__main__":
    main()
