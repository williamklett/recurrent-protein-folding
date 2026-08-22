#!/usr/bin/env python3
"""Track how the *meaning* of an activation changes between trunk blocks.

All comparisons are done in the COMMON, layer-invariant distogram space (the one
space whose semantics don't drift).  For each block t we split the total belief
change exactly into a CONTENT term (decoder fixed, activation moves) and a
REINTERPRETATION term (activation fixed at z_{t+1}, decoder moves t->t+1):

    P_{t+1}(z_{t+1}) - P_t(z_t)
        = [P_t(z_{t+1}) - P_t(z_t)]            # content     (probe_t, z_t -> z_{t+1})
        + [P_{t+1}(z_{t+1}) - P_t(z_{t+1})]    # reinterp    (probe_t vs probe_{t+1} on z_{t+1})

P_t = expected-distance map decoded by probe_t  (Angstrom, interpretable).
Convention (matches energy_decomp.py): probe_t reads z_t = input to block t.

Also: a SMOOTHNESS CONTROL.  For a frozen target state z_t we sweep older
decoders P_{t-k}(z_t), k=0..K, and measure RMS deviation from P_t(z_t).  Smooth
growth => structured semantic drift; a cliff => mere OOD degradation.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding
from recurrent_folding.esmfold_probes import PairDistogramBottleneck

SEQ_SEP = 3
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
    for q in pr.parameters():
        q.requires_grad = False
    return pr


def decode(probe, z, i_t, j_t, L, cen):
    """Return (expected-distance map [L,L], probs [L,L,64]) for probe applied to z."""
    logits = probe.encode(z.reshape(-1, z.shape[-1]), i_t, j_t, L).reshape(1, L, L, 64)
    logits = 0.5 * (logits + logits.transpose(1, 2))
    probs = F.softmax(logits, -1)
    ed = (probs * cen).sum(-1)[0]
    return ed, probs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
    ap.add_argument("--out-csv", default="runs/meaning_drift_ubq.csv")
    ap.add_argument("--kmax", type=int, default=5)
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
    keep = ((jj - ii).abs() > SEQ_SEP)
    nkeep = float(keep.sum())

    def rms(dmap):
        return float(((dmap ** 2) * keep).sum() / nkeep) ** 0.5  # Angstrom RMS over kept pairs

    def js(p, q):
        m = 0.5 * (p + q)
        kl = lambda a, b: (a * (torch.log(a + 1e-9) - torch.log(b + 1e-9))).sum(-1)
        d = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float((d * keep).sum() / nkeep)

    # capture z_t = input to block t for all t
    init = {}
    def hook(_m, a):
        if "z" not in init:
            init["s"] = a[0].detach(); init["z"] = a[1].detach()
    h = model.trunk.blocks[0].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(**inputs, num_recycles=0)
    h.remove()
    residx = torch.arange(L, device=device).unsqueeze(0)

    zs = [init["z"]]
    s_s, s_z = init["s"], init["z"]
    with torch.no_grad():
        for t in range(48):
            s_s, s_z = model.trunk.blocks[t](s_s, s_z, mask=None, residue_index=residx)
            zs.append(s_z.detach())
    pair_dim = zs[0].shape[-1]

    probes = {t: load_probe(t, device, pair_dim) for t in range(48)}

    # ---- content vs reinterpretation, per block ----
    rows = []
    with torch.no_grad():
        for t in range(1, 47):
            pt, pt1 = probes.get(t), probes.get(t + 1)
            if pt is None or pt1 is None:
                continue
            ed_t_zt, _ = decode(pt, zs[t], i_t, j_t, L, cen)         # P_t(z_t)      in-dist
            ed_t_zt1, pr_t_zt1 = decode(pt, zs[t + 1], i_t, j_t, L, cen)   # P_t(z_{t+1})  1-step OOD
            ed_t1_zt1, pr_t1_zt1 = decode(pt1, zs[t + 1], i_t, j_t, L, cen)# P_{t+1}(z_{t+1}) in-dist
            content = ed_t_zt1 - ed_t_zt
            reinterp = ed_t1_zt1 - ed_t_zt1
            total = ed_t1_zt1 - ed_t_zt
            rows.append({
                "layer": t,
                "rms_content": rms(content),
                "rms_reinterp": rms(reinterp),
                "rms_total": rms(total),
                "js_reinterp": js(pr_t1_zt1, pr_t_zt1),
                "frac_reinterp": rms(reinterp) / (rms(reinterp) + rms(content) + 1e-9),
            })
            r = rows[-1]
            print(f"L{t:02d} | content {r['rms_content']:6.3f} A | reinterp {r['rms_reinterp']:6.3f} A "
                  f"| total {r['rms_total']:6.3f} A | reinterp-frac {r['frac_reinterp']:.2f} "
                  f"| JS_reinterp {r['js_reinterp']:.4f}", flush=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out_csv}", flush=True)

    def blk(lo, hi):
        c = np.mean([r["rms_content"] for r in rows if lo <= r["layer"] <= hi])
        rr = np.mean([r["rms_reinterp"] for r in rows if lo <= r["layer"] <= hi])
        return c, rr
    for name, lo, hi in [("EARLY", 1, 11), ("MID", 12, 31), ("LATE", 32, 46)]:
        c, rr = blk(lo, hi)
        print(f"[{name} L{lo}-{hi}] mean content {c:.3f} A | mean reinterp {rr:.3f} A | reinterp/content {rr/c:.2f}", flush=True)

    # ---- smoothness control: older decoders on a frozen state ----
    print("\n=== SMOOTHNESS CONTROL: P_{t-k}(z_t) RMS deviation (A) from P_t(z_t), averaged over t ===", flush=True)
    K = args.kmax
    targets = [t for t in range(K + 1, 47) if probes.get(t) is not None]
    with torch.no_grad():
        avg = []
        for k in range(K + 1):
            devs = []
            for t in targets:
                ptk = probes.get(t - k)
                if ptk is None:
                    continue
                ed_native, _ = decode(probes[t], zs[t], i_t, j_t, L, cen)
                ed_old, _ = decode(ptk, zs[t], i_t, j_t, L, cen)
                devs.append(rms(ed_old - ed_native))
            avg.append(np.mean(devs) if devs else float("nan"))
            print(f"k={k} | mean RMS dev {avg[-1]:.3f} A   (decoder is {k} layers stale)", flush=True)
    # per-step increments => smooth if roughly constant, cliff if first step dominates
    inc = [avg[k] - avg[k - 1] for k in range(1, K + 1)]
    print(f"per-step increments (A): {['%.3f' % d for d in inc]}", flush=True)


if __name__ == "__main__":
    main()
