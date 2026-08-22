#!/usr/bin/env python3
"""Fit per-layer linear TRANSPORT so the FIXED native ESMFold distogram head
reproduces each layer's faithful (in-distribution) belief.

Motivation (the "each layer has its own dictionary" hypothesis):
  The native head is a single linear map H(z)_ij = W z_ij + b  (W in R^{64x128}).
  It is only in-distribution for the FINAL trunk output z_47.  At an intermediate
  layer t, the same activation geometry means something different -> applying H
  directly (the "logit lens") is wrong.  The per-layer probe P_t is the faithful
  readout (trained in-distribution, anchored to H at the end).

  If the representational drift between layers is (approximately) a LINEAR change
  of basis, then there exists a transport A_t (128x128) such that H(A_t z_t) ~
  P_t(z_t).  Because H is linear, H(A_t z) = (W A_t) z + b, so the effective
  readout is M_t = W A_t in R^{64x128}.  We therefore fit M_t (and bias d_t)
  DIRECTLY to reproduce the probe belief, then recover the minimum-distortion
  transport A_t = I + W^+ (M_t - W).  Distortion = ||M_t - W|| / ||W||.

  CAVEAT we measure honestly: the probe reads z PLUS 2 separation features
  (|i-j|/(L-1), signed); the native head sees only z.  So a pure-in-z map M_t z
  cannot capture sep-dependence or probe nonlinearity -> that irreducible part
  shows up as residual even for the unconstrained variant (a).

Three nested-constraint variants, all fit on TRAIN proteins, evaluated on
HELD-OUT proteins (cross-protein generalization -- the real test):
  (a) unconstrained M_t                       -> linear-in-z ceiling
  (b) ridge toward native:  M_t = W + dM, penalize ||dM||_F^2
  (c) near-identity transport: penalize ||A_t - I||_F^2 = ||W^+ dM||_F^2
Baseline floor: 'native' = untuned logit lens (W z + b), no fit.

Target = probe belief, symmetrized over (i,j) exactly as the native head and the
probe trajectory are symmetrized.  Loss = mean KL(belief || pred).

Splits:
  TRAIN/TEST   : the 20 benchmark_* proteins (probe in-distribution), protein-level.
  OOD          : synthetic_* / hard_synthetic_* proteins the probe never saw.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from recurrent_folding.esmfold_probes import PairDistogramBottleneck

ROOT = Path.home() / "recurrent-protein-folding"
PROBE_DIR = ROOT / "runs" / "general_belief_replay_benchmark20_cw30"
SEQ_SEP = 3
S_DEFAULT = 3000

BENCH = [  # the 20 benchmark proteins the probes were trained on
    "1AO6","7ACN","8fmi","T1104","T1120","T1122","T1124","T1132","T1137s4",
    "T1137s6","T1147","T1155","T1159","T1170","T1173","T1176","T1183","T1194",
    "mbn","ubq",
]


def centers(device):
    s, e, b = 2.3125, 21.6875, 64
    w = (e - s) / b
    return torch.linspace(s + w / 2, e - w / 2, b, device=device)


def load_native_head(device):
    from transformers import EsmForProteinFolding
    print("  loading ESMFold to extract native distogram head...", flush=True)
    m = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    head = None
    if hasattr(m, "distogram_head"):
        head = m.distogram_head
    else:
        for n, mod in m.named_modules():
            if n.endswith("distogram_head"):
                head = mod; break
    W = head.weight.detach().to(device, torch.float32).clone()   # (64,128)
    b = head.bias.detach().to(device, torch.float32).clone()     # (64,)
    del m
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  native head W {tuple(W.shape)} b {tuple(b.shape)}", flush=True)
    return W, b


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


def find_dir(name):
    """Map a short protein name to its activations dir under runs/."""
    cands = [
        ROOT / "runs" / f"benchmark_{name}",
        ROOT / "runs" / f"synthetic_design_{name}",
        ROOT / "runs" / f"hard_synthetic_miss_{name}",
    ]
    for c in cands:
        if (c / "esmfold_activations.pt").exists():
            return c
    return None


def sample_pairs(L, S, rng):
    """Sample up to S unordered pairs (i<j) with |i-j| > SEQ_SEP."""
    ii, jj = np.triu_indices(L, k=SEQ_SEP + 1)
    if len(ii) == 0:
        return None, None
    if len(ii) > S:
        sel = rng.choice(len(ii), size=S, replace=False)
        ii, jj = ii[sel], jj[sel]
    return ii, jj


@torch.no_grad()
def build_bucket(dirs, probes, device, S, seed, label):
    """For each protein and layer, sample pairs, compute symmetrized probe belief
    (target) and symmetrized z (input to native/transport map).
    Returns dict layer -> (z_sym [N,128] fp32 cpu, target_probs [N,64] fp32 cpu)."""
    rng = np.random.default_rng(seed)
    cen = centers(device)
    acc = {t: [[], []] for t in range(1, 47)}
    for name, d in dirs:
        t0 = time.time()
        blob = torch.load(d / "esmfold_activations.pt", map_location="cpu", weights_only=False)
        z_full = blob["z"]  # (48,1,L,L,128) fp16
        L = z_full.shape[2]
        ii, jj = sample_pairs(L, S, rng)
        if ii is None:
            print(f"    [{label}] {name} L={L} too short, skip", flush=True)
            continue
        it = torch.as_tensor(ii, device=device)
        jt = torch.as_tensor(jj, device=device)
        for t in range(1, 47):
            pr = probes.get(t)
            if pr is None:
                continue
            zt = z_full[t, 0].to(device, torch.float32)  # (L,L,128)
            zij = zt[it, jt]            # (N,128)
            zji = zt[jt, it]            # (N,128)
            z_sym = 0.5 * (zij + zji)
            # symmetrized probe belief (matches trajectory convention)
            lij = pr.encode(zij, it, jt, L)        # (N,64)
            lji = pr.encode(zji, jt, it, L)
            ltgt = 0.5 * (lij + lji)
            tgt = F.softmax(ltgt, dim=-1)
            acc[t][0].append(z_sym.cpu())
            acc[t][1].append(tgt.cpu())
            del zt
        print(f"    [{label}] {name} L={L} npairs={len(ii)} ({time.time()-t0:.1f}s)", flush=True)
        del z_full, blob
    out = {}
    for t in range(1, 47):
        if acc[t][0]:
            out[t] = (torch.cat(acc[t][0]), torch.cat(acc[t][1]))
    return out


def kl(target, pred_log):
    # KL(target || pred), target probs, pred_log = log-softmax of pred logits
    return (target * (torch.log(target + 1e-9) - pred_log)).sum(-1).mean()


def ed_rms(pred_probs, target_probs, cen):
    ed_p = (pred_probs * cen).sum(-1)
    ed_t = (target_probs * cen).sum(-1)
    return float(((ed_p - ed_t) ** 2).mean() ** 0.5)


def fit_variant(Z, T, W, b, Wpinv, variant, lam, device, steps, lr):
    """Z (N,128), T (N,64) on device. Fit M,d minimizing KL + penalty."""
    M = W.clone().requires_grad_(True)
    d = b.clone().requires_grad_(True)
    opt = torch.optim.Adam([M, d], lr=lr)
    Wn2 = (W ** 2).sum()
    for _ in range(steps):
        opt.zero_grad()
        logit = Z @ M.t() + d
        loss = kl(T, F.log_softmax(logit, dim=-1))
        if variant == "b":
            loss = loss + lam * ((M - W) ** 2).sum() / Wn2
        elif variant == "c":
            A_dev = Wpinv @ (M - W)          # (128,128) deviation of transport from I
            loss = loss + lam * (A_dev ** 2).sum()
        loss.backward()
        opt.step()
    return M.detach(), d.detach()


@torch.no_grad()
def evaluate(M, d, Z, T, cen):
    logit = Z @ M.t() + d
    logp = F.log_softmax(logit, dim=-1)
    return float(kl(T, logp)), ed_rms(logp.exp(), T, cen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=S_DEFAULT)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lam-b", type=float, default=3e-2)
    ap.add_argument("--lam-c", type=float, default=3e-2)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--n-ood", type=int, default=12)
    ap.add_argument("--out-csv", default=str(ROOT / "runs" / "transport_fit.csv"))
    ap.add_argument("--out-npz", default=str(ROOT / "runs" / "transport_fit.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cen = centers(device)

    # ---- split benchmark proteins train/test ----
    rng = np.random.default_rng(args.seed)
    bench = [n for n in BENCH if find_dir(n) is not None]
    perm = rng.permutation(len(bench))
    test_names = [bench[i] for i in perm[: args.n_test]]
    train_names = [bench[i] for i in perm[args.n_test:]]
    train_dirs = [(n, find_dir(n)) for n in train_names]
    test_dirs = [(n, find_dir(n)) for n in test_names]
    # ood: synthetic proteins (never seen by probe)
    all_ood = []
    for d in sorted((ROOT / "runs").glob("synthetic_design_*")) + \
             sorted((ROOT / "runs").glob("hard_synthetic_miss_*")):
        if (d / "esmfold_activations.pt").exists():
            all_ood.append((d.name, d))
    oo = rng.permutation(len(all_ood))[: args.n_ood]
    ood_dirs = [all_ood[i] for i in oo]

    print(f"TRAIN {len(train_dirs)}: {train_names}", flush=True)
    print(f"TEST  {len(test_dirs)}: {test_names}", flush=True)
    print(f"OOD   {len(ood_dirs)}: {[n for n,_ in ood_dirs]}", flush=True)

    W, b = load_native_head(device)
    Wpinv = torch.linalg.pinv(W)            # (128,64)
    Wn = float((W ** 2).sum() ** 0.5)
    probes = {t: load_probe(t, device) for t in range(1, 47)}
    nprobe = sum(v is not None for v in probes.values())
    print(f"loaded {nprobe} probes", flush=True)

    print("building TRAIN bucket...", flush=True)
    train_b = build_bucket(train_dirs, probes, device, args.S, args.seed + 1, "train")
    print("building TEST bucket...", flush=True)
    test_b = build_bucket(test_dirs, probes, device, args.S, args.seed + 2, "test")
    print("building OOD bucket...", flush=True)
    ood_b = build_bucket(ood_dirs, probes, device, args.S, args.seed + 3, "ood")

    rows = []
    M_store = {"a": {}, "b": {}}
    print("\n=== per-layer fit ===", flush=True)
    print(f"{'L':>3} | {'native_test':>11} {'a_test':>8} {'b_test':>8} {'c_test':>8} "
          f"| {'a_ood':>7} {'b_ood':>7} | {'distort_a':>9} {'distort_b':>9}", flush=True)
    for t in range(1, 47):
        if t not in train_b or t not in test_b:
            continue
        Ztr, Ttr = train_b[t][0].to(device), train_b[t][1].to(device)
        Zte, Tte = test_b[t][0].to(device), test_b[t][1].to(device)
        has_ood = t in ood_b
        if has_ood:
            Zood, Tood = ood_b[t][0].to(device), ood_b[t][1].to(device)

        # native baseline (logit lens), no fit
        nat_te = evaluate(W, b, Zte, Tte, cen)
        nat_ood = evaluate(W, b, Zood, Tood, cen) if has_ood else (float("nan"),) * 2

        res = {}
        dist = {}
        for v, lam in [("a", 0.0), ("b", args.lam_b), ("c", args.lam_c)]:
            M, d = fit_variant(Ztr, Ttr, W, b, Wpinv, v, lam, device, args.steps, args.lr)
            tr = evaluate(M, d, Ztr, Ttr, cen)
            te = evaluate(M, d, Zte, Tte, cen)
            od = evaluate(M, d, Zood, Tood, cen) if has_ood else (float("nan"),) * 2
            res[v] = (tr, te, od)
            dist[v] = float(((M - W) ** 2).sum() ** 0.5) / Wn
            A_dev = float((((Wpinv @ (M - W)) ** 2).sum()) ** 0.5)  # ||A - I||_F
            if v in ("a", "b"):
                M_store[v][t] = (M.cpu().numpy(), d.cpu().numpy())
            rows.append({
                "layer": t, "variant": v,
                "train_kl": res[v][0][0], "train_ed_rms": res[v][0][1],
                "test_kl": te[0], "test_ed_rms": te[1],
                "ood_kl": od[0], "ood_ed_rms": od[1],
                "distortion": dist[v], "transport_dev": A_dev,
            })
        rows.append({
            "layer": t, "variant": "native",
            "train_kl": float("nan"), "train_ed_rms": float("nan"),
            "test_kl": nat_te[0], "test_ed_rms": nat_te[1],
            "ood_kl": nat_ood[0], "ood_ed_rms": nat_ood[1],
            "distortion": 0.0, "transport_dev": 0.0,
        })
        print(f"{t:>3} | {nat_te[0]:>11.4f} {res['a'][1][0]:>8.4f} {res['b'][1][0]:>8.4f} "
              f"{res['c'][1][0]:>8.4f} | {res['a'][2][0]:>7.4f} {res['b'][2][0]:>7.4f} "
              f"| {dist['a']:>9.3f} {dist['b']:>9.3f}", flush=True)
        del Ztr, Ttr, Zte, Tte
        if has_ood:
            del Zood, Tood

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {args.out_csv}", flush=True)

    # save fitted maps for downstream single-energy-landscape analysis
    layers_a = sorted(M_store["a"].keys())
    np.savez(
        args.out_npz,
        W=W.cpu().numpy(), b=b.cpu().numpy(),
        layers=np.array(layers_a),
        M_a=np.stack([M_store["a"][t][0] for t in layers_a]),
        d_a=np.stack([M_store["a"][t][1] for t in layers_a]),
        M_b=np.stack([M_store["b"][t][0] for t in layers_a]),
        d_b=np.stack([M_store["b"][t][1] for t in layers_a]),
    )
    print(f"Wrote {args.out_npz}", flush=True)

    # ---- block summaries (expected-distance RMS to faithful belief, test split) ----
    def blk(variant, lo, hi, col):
        vals = [r[col] for r in rows if r["variant"] == variant and lo <= r["layer"] <= hi
                and not np.isnan(r[col])]
        return float(np.mean(vals)) if vals else float("nan")
    print("\n=== block-mean TEST expected-distance RMS (A) to faithful belief ===", flush=True)
    print(f"{'block':>10} | {'native':>7} {'(a)uncon':>9} {'(b)ridge':>9} {'(c)near-I':>9}", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-46", 32, 46)]:
        print(f"{name:>10} | {blk('native',lo,hi,'test_ed_rms'):>7.3f} "
              f"{blk('a',lo,hi,'test_ed_rms'):>9.3f} {blk('b',lo,hi,'test_ed_rms'):>9.3f} "
              f"{blk('c',lo,hi,'test_ed_rms'):>9.3f}", flush=True)
    print("\n=== block-mean OOD expected-distance RMS (A) to faithful belief ===", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-46", 32, 46)]:
        print(f"{name:>10} | {blk('native',lo,hi,'ood_ed_rms'):>7.3f} "
              f"{blk('a',lo,hi,'ood_ed_rms'):>9.3f} {blk('b',lo,hi,'ood_ed_rms'):>9.3f} "
              f"{blk('c',lo,hi,'ood_ed_rms'):>9.3f}", flush=True)
    print("\n=== block-mean distortion ||M-W||/||W|| ===", flush=True)
    for name, lo, hi in [("EARLY 1-11", 1, 11), ("MID 12-31", 12, 31), ("LATE 32-46", 32, 46)]:
        print(f"{name:>10} | (a) {blk('a',lo,hi,'distortion'):.3f}  "
              f"(b) {blk('b',lo,hi,'distortion'):.3f}", flush=True)


if __name__ == "__main__":
    main()
