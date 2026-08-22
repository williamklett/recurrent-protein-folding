#!/usr/bin/env python3
"""Natural-gradient cosine alignment test.

Question: is the trunk doing *natural* gradient descent on the probe-read LJ
energy?  I.e. does the block update Δz_t align with the NATURAL gradient
−M(z_t)⁻¹ ∇_{z_t} E  far better than with the raw −∇_{z_t} E (cos≈0.05)?

M(z_t) = Fisher information of the probe's distogram likelihood w.r.t. z_t:
    M = Jᵀ F_cat J ,   J = ∂logits/∂z_t ,
    F_cat = block-diag per pair of  (diag(p) − p pᵀ)   (categorical Fisher).
We never form M.  We solve  (M + λI) ng = ∇E  by conjugate gradient using
Fisher-vector products  Mv = Jᵀ F_cat (J v)  via forward (JVP, double-backward)
and reverse (VJP) autograd through one trunk block.  λ = Hutchinson-scaled
damping for a well-posed solve.

Outputs per layer:  cos(Δz,−∇E) [raw],  cos(Δz,−M⁻¹∇E) [natural],  plus the
same for the self-belief gradient as a sanity reference.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import PairDistogramBottleneck

SIGMA = 6.8
EPS = 1.0
SEQ_SEP = 3
PROBE_DIR = Path("runs/general_belief_replay_benchmark20_cw30")

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def extract_sequence_and_ca(pdb_path: Path, chain_hint: str):
    from collections import Counter
    counts: Counter = Counter()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
                counts[ln[21]] += 1
    if not counts:
        raise ValueError(f"no CA atoms in {pdb_path}")
    hint = chain_hint.strip() or " "
    chain = hint if hint in counts and counts[hint] >= 20 else counts.most_common(1)[0][0]
    seq, coords, seen = [], [], set()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21] == chain:
                rs = (int(ln[22:26]), ln[26])
                if rs in seen:
                    continue
                seen.add(rs)
                seq.append(AA3.get(ln[17:20].strip(), "X"))
                coords.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return "".join(seq), np.asarray(coords, dtype=np.float32), chain


def get_centers(device):
    start, end, bins = 2.3125, 21.6875, 64
    width = (end - start) / bins
    return torch.linspace(start + width / 2, end - width / 2, bins, device=device)


def cos(a, b):
    a = a.detach().flatten()
    b = b.detach().flatten()
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))


def load_probe(layer, device, pair_dim=128):
    path = PROBE_DIR / f"layer_{layer:02d}" / "probe.pt"
    if not path.exists():
        return None
    probe = PairDistogramBottleneck(pair_dim=pair_dim, bins=64, hidden_dim=256).to(device)
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False
    return probe


def jvp(out, inp, v):
    """J v where J = d out / d inp, via double-backward."""
    u = torch.zeros_like(out, requires_grad=True)
    (g,) = torch.autograd.grad(out, inp, grad_outputs=u, create_graph=True, retain_graph=True)
    (Jv,) = torch.autograd.grad(g, u, grad_outputs=v, retain_graph=True)
    return Jv


def natural_grad(grad, logits, probs, z_in, damping_rel, maxiter, tol):
    """Solve (M + λI) ng = grad  by CG;  M = Jᵀ F_cat J,  J = d logits / d z_in."""
    def fisher_vp(v, lam):
        Jv = jvp(logits, z_in, v)                       # ∂logits in direction v
        Fu = probs * Jv - probs * (probs * Jv).sum(-1, keepdim=True)  # F_cat (Jv)
        (JtFu,) = torch.autograd.grad(logits, z_in, grad_outputs=Fu, retain_graph=True)
        return JtFu + lam * v

    # Hutchinson trace estimate -> scale-appropriate damping λ.
    r = torch.randn_like(grad)
    Mr = fisher_vp(r, 0.0)
    tr = float((r * Mr).sum() / (r * r).sum())
    lam = max(damping_rel * abs(tr), 1e-8)

    b = grad.detach()
    x = torch.zeros_like(b)
    rk = b.clone()
    pk = rk.clone()
    rs = float((rk * rk).sum())
    b_norm = float((b * b).sum()) + 1e-30
    for _ in range(maxiter):
        Ap = fisher_vp(pk, lam)
        denom = float((pk * Ap).sum()) + 1e-30
        alpha = rs / denom
        x = x + alpha * pk
        rk = rk - alpha * Ap
        rs_new = float((rk * rk).sum())
        if rs_new / b_norm < tol * tol:
            break
        pk = rk + (rs_new / rs) * pk
        rs = rs_new
    return x.detach(), lam


def run(sequence, model, tokenizer, device, gt_coords, damping_rel, maxiter, tol):
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(device)
    centers = get_centers(device)
    L = len(sequence)
    i_idx, j_idx = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    i_t, j_t = i_idx.flatten(), j_idx.flatten()
    sep = (j_idx - i_idx).abs()
    keep = (sep > SEQ_SEP).float()

    with torch.no_grad():
        final_probs = F.softmax(model(**inputs, num_recycles=0).distogram_logits, dim=-1).detach()

    init = {}
    def hook(_m, args):
        if "s" not in init:
            init["s"] = args[0].detach(); init["z"] = args[1].detach()
    h = model.trunk.blocks[0].register_forward_pre_hook(hook)
    with torch.no_grad():
        model(**inputs, num_recycles=0)
    h.remove()

    s_s, s_z = init["s"], init["z"]
    residx = torch.arange(L, device=device).unsqueeze(0)
    rows = []

    for t in range(48):
        probe = load_probe(t + 1, device, pair_dim=s_z.shape[-1])
        if probe is None:
            with torch.no_grad():
                s_s, s_z = model.trunk.blocks[t](s_s, s_z, mask=None, residue_index=residx)
            continue

        z_in = s_z.detach().clone().requires_grad_(True)
        s_in = s_s.detach().clone()
        s_out, z_out = model.trunk.blocks[t](s_in, z_in, mask=None, residue_index=residx)

        logits = probe.encode(z_out.reshape(-1, s_z.shape[-1]), i_t, j_t, L).reshape(1, L, L, 64)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        probs = F.softmax(logits, dim=-1)

        ed = (probs * centers).sum(-1)
        d = ed.clamp(min=0.5)
        rr = SIGMA / d
        lj = (4 * EPS * (rr ** 12 - rr ** 6) * keep).sum() / L

        log_probs = F.log_softmax(logits, dim=-1)
        bel = -(final_probs * log_probs).sum(-1).mean()

        grad_lj = torch.autograd.grad(lj, z_in, retain_graph=True)[0]
        grad_bel = torch.autograd.grad(bel, z_in, retain_graph=True)[0]

        ng_lj, lam = natural_grad(grad_lj, logits, probs, z_in, damping_rel, maxiter, tol)
        ng_bel, _ = natural_grad(grad_bel, logits, probs, z_in, damping_rel, maxiter, tol)

        dz = (z_out - z_in).detach()

        # --- de-diluted alignment in the Fisher metric (removes don't-care dims) ---
        # quad_M(v) = (J v)^T F_cat (J v)  with UNDAMPED M (active subspace only).
        D = dz.numel()
        def quad_M(v):
            Jv = jvp(logits, z_in, v)
            pv = (probs * Jv).sum(-1)
            return float((probs * Jv ** 2).sum() - (pv ** 2).sum())
        quad_dz = quad_M(dz)

        def cosM_descent(direction, q_dir, grad, ng):
            # cos(direction, -ng) in the M-metric = -(direction·g)/sqrt(qdir · g·ng)
            gMg = float((grad * ng).sum())                 # ||ng||_M^2 = g^T M^-1 g
            den = (q_dir * gMg) ** 0.5
            return -float((direction * grad).sum()) / den if den > 1e-20 else 0.0

        # in-metric random baseline (correct noise floor for cosM): mean over 3 draws
        rfloor = []
        for _ in range(3):
            r = torch.randn_like(dz)
            rfloor.append(cosM_descent(r, quad_M(r), grad_lj, ng_lj))
        cosM_rand = float(np.mean(rfloor))

        row = {
            "layer": t,
            "energy_next": float(lj.detach()),
            "cos_desc_lj": cos(dz, -grad_lj),              # raw descent, Euclidean (diluted)
            "cos_desc_nat": cos(dz, -ng_lj),               # natural-dir descent, Euclidean (diluted)
            "cosM_desc_lj": cosM_descent(dz, quad_dz, grad_lj, ng_lj),   # Fisher descent (de-diluted) KEY
            "cosM_desc_bel": cosM_descent(dz, quad_dz, grad_bel, ng_bel),
            "cosM_random": cosM_rand,                      # in-metric noise floor
            "cos_random_euclid": cos(dz, torch.randn_like(dz)),
            "floor_euclid": 1.0 / (D ** 0.5),
            "quad_dz_M": quad_dz,
            "norm_dz": float(dz.norm()),
            "lambda": lam,
        }
        rows.append(row)
        print(f"L{t:02d} | E {float(lj):6.2f} | euclid desc {row['cos_desc_lj']:+.4f} "
              f"| FISHER desc lj {row['cosM_desc_lj']:+.4f} bel {row['cosM_desc_bel']:+.4f} "
              f"| Mrand {row['cosM_random']:+.4f}", flush=True)

        s_s = s_out.detach(); s_z = z_out.detach()

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=str,
                    default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
    ap.add_argument("--pdb", type=str, default="data/1UBQ.pdb")
    ap.add_argument("--out-csv", type=str, default="runs/cosine_natural_grad.csv")
    ap.add_argument("--damping-rel", type=float, default=1e-2)
    ap.add_argument("--cg-iters", type=int, default=20)
    ap.add_argument("--cg-tol", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    gt = None
    if os.path.exists(args.pdb):
        seq, coords, _ = extract_sequence_and_ca(Path(args.pdb), "")
        args.sequence = seq
        gt = torch.from_numpy(coords).float()

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True

    print(f"Running natural-gradient test (L={len(args.sequence)}, damping_rel={args.damping_rel}, "
          f"cg_iters={args.cg_iters})", flush=True)
    rows = run(args.sequence, model, tok, device, gt, args.damping_rel, args.cg_iters, args.cg_tol)
    if not rows:
        print("No results."); return

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.out_csv}", flush=True)

    def stat(col, lo=0):
        v = [r[col] for r in rows if abs(r[col]) <= 1.5 and r["layer"] >= lo]
        return (np.mean(v), np.median(v), np.max(v), np.std(v), len(v)) if v else (0, 0, 0, 0, 0)
    floor = rows[0]["floor_euclid"]
    print(f"\nD (hidden dims) = {int((1.0/floor)**2)} ; euclid floor 1/sqrt(D) = {floor:.5f}", flush=True)
    rm = np.mean([abs(r["cosM_random"]) for r in rows]); rs = np.std([r["cosM_random"] for r in rows])
    print(f"in-metric (Fisher) random baseline: |cosM_random| mean {rm:.4f}, std {rs:.4f}  <- correct floor for cosM", flush=True)
    for lo in (0, 14):
        for col in ("cos_desc_lj", "cosM_desc_lj", "cosM_desc_bel"):
            m, md, x, sd, n = stat(col, lo)
            print(f"[layers>={lo:2d}] {col:13s} mean {m:+.4f} median {md:+.4f} max {x:+.4f} std {sd:.4f} (n={n})", flush=True)


if __name__ == "__main__":
    main()
