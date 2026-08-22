#!/usr/bin/env python3
"""Cosine alignment between trunk update Δz_t and −∇_z LJ(E[d(z_t)]).

If the trunk is implementing gradient descent on the LJ energy of the
expected shape, cos(Δz_t, −∇LJ) should be systematically positive,
much larger than a random-direction baseline.

We also compare against the GT-aligned loss and the layer-47 probe loss
to put the LJ alignment in context (these were the existing reference points
in cosine_alignment.py).
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from recurrent_folding.esmfold_probes import PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers

ESMFOLD_DISTO_MIN = 2.3125
ESMFOLD_DISTO_MAX = 21.6875
SIGMA = 6.8
EPS   = 1.0
SEQ_SEP = 3

ROOT       = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "runs/general_belief_replay_benchmark20_cw30"
RUNS_DIR   = ROOT / "runs"


def soft_bin_targets(D):
    Dc = D.clamp(ESMFOLD_DISTO_MIN, ESMFOLD_DISTO_MAX - 1e-3)
    binw = (ESMFOLD_DISTO_MAX - ESMFOLD_DISTO_MIN) / ESMFOLD_DISTO_BINS
    idx = ((Dc - ESMFOLD_DISTO_MIN) / binw).long()
    return idx.clamp(0, ESMFOLD_DISTO_BINS - 1).flatten()


def cosine(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) < 1e-12 or float(nb) < 1e-12: return 0.0
    return float((a * b).sum() / (na * nb))


def load_probe(layer, pair_dim, device):
    path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(pair_dim=pair_dim, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters(): p.requires_grad_(False)
    return pr


def kabsch_tm(pred, gt):
    n = min(len(pred), len(gt))
    p, g = pred[:n].copy(), gt[:n].copy()
    p -= p.mean(0); gc = g - g.mean(0)
    H = p.T @ gc; U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1,1,np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    pa = p @ R.T + g.mean(0)
    d = np.sqrt(((pa - g)**2).sum(-1))
    d0 = max(0.5, 1.24*np.cbrt(max(n-15,1)) - 1.8)
    return float(np.mean(1/(1+(d/d0)**2)))


def compute_for_protein(name, act_p, gt_p, pred_p, device):
    cache  = torch.load(act_p, map_location="cpu", weights_only=False)
    z_all  = cache["z"].squeeze(1).float()    # (48, L, L, 128)
    n_layers = z_all.shape[0]
    L  = z_all.shape[1]
    pd = z_all.shape[-1]

    gt = np.load(gt_p)
    D_gt_np = gt["distances"]
    n = min(D_gt_np.shape[0], L)
    if n != L:
        print(f"  {name}: length mismatch, skip"); return []
    D_gt = torch.from_numpy(D_gt_np[:L, :L]).float().to(device)
    gt_targets = soft_bin_targets(D_gt).to(device)

    pred_ca = np.load(pred_p)["pred_ca"].astype(float)
    gt_ca   = gt["ca_coords"].astype(float)
    tm      = kabsch_tm(pred_ca, gt_ca)

    i_idx, j_idx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    i_t = i_idx.flatten().to(device).long()
    j_t = j_idx.flatten().to(device).long()
    sep = (j_t - i_t).abs()
    keep_mask = (sep > SEQ_SEP)

    centers_full = distogram_bin_centers(device, torch.float32).reshape(-1)
    centers = centers_full[:ESMFOLD_DISTO_BINS]

    # Reference targets: layer-47 probe's soft distogram (probe-trained equivalent)
    probe47_dir = PROBES_DIR / "layer_47"
    probe47_path = probe47_dir / "probe.pt"
    p47 = None
    if probe47_path.exists():
        probe47 = load_probe(47, pd, device)
        with torch.no_grad():
            z47 = z_all[-1].to(device)
            logits47 = probe47.encode(z47.reshape(-1, pd), i_t, j_t, L)
            p47 = F.softmax(logits47, dim=-1).detach()
        del probe47

    rows = []
    rng = torch.Generator(device=device).manual_seed(0)

    for layer in range(1, n_layers):
        probe_path = PROBES_DIR / f"layer_{layer:02d}" / "probe.pt"
        if not probe_path.exists(): continue
        probe = load_probe(layer, pd, device)

        z_cur = z_all[layer - 1].to(device)
        z_next = z_all[layer].to(device)
        delta_z = (z_next - z_cur).flatten()
        if float(delta_z.norm()) < 1e-9:
            del probe; continue

        def make_grad(loss_fn):
            z = z_cur.detach().clone().requires_grad_(True)
            logits = probe.encode(z.reshape(-1, pd), i_t, j_t, L)
            loss   = loss_fn(logits)
            return torch.autograd.grad(loss, z)[0].flatten()

        def loss_lj(logits):
            probs = F.softmax(logits, dim=-1)
            ed    = (probs * centers).sum(-1)
            d     = ed.clamp(min=0.5)
            r     = SIGMA / d
            lj    = 4 * EPS * (r**12 - r**6)
            # mask to keep |i-j| > SEQ_SEP only
            return (lj * keep_mask.float()).sum() / L

        def loss_gt(logits):
            return F.cross_entropy(logits, gt_targets, reduction="mean")

        def loss_final(logits):
            logp = F.log_softmax(logits, dim=-1)
            return -(p47 * logp).sum(dim=-1).mean() if p47 is not None else logits.sum() * 0

        g_lj    = make_grad(loss_lj)
        g_gt    = make_grad(loss_gt)
        g_final = make_grad(loss_final) if p47 is not None else None

        rand_dir = torch.randn(delta_z.shape, device=device, generator=rng)

        rows.append({
            "protein": name, "tm": tm, "layer": layer,
            "cos_neg_lj":    cosine(delta_z, -g_lj),
            "cos_neg_gt":    cosine(delta_z, -g_gt),
            "cos_neg_final": cosine(delta_z, -g_final) if g_final is not None else float("nan"),
            "cos_random":    cosine(delta_z, rand_dir),
        })
        del probe
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path, default=ROOT / "runs/cosine_lj.csv")
    ap.add_argument("--out-png", type=Path, default=ROOT / "runs/cosine_lj.png")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proteins = []
    for act_p in sorted(RUNS_DIR.glob("*/esmfold_activations.pt")):
        run_dir = act_p.parent
        gt_p   = run_dir / "gt_ca_distances.npz"
        pred_p = next(run_dir.glob("**/esmfold_prediction_ca.npz"), None)
        if gt_p.exists() and pred_p:
            proteins.append((run_dir.name, act_p, gt_p, pred_p))

    print(f"Found {len(proteins)} proteins")
    all_rows = []
    for pi, (name, act_p, gt_p, pred_p) in enumerate(proteins):
        print(f"[{pi+1}/{len(proteins)}] {name}", flush=True)
        rs = compute_for_protein(name, act_p, gt_p, pred_p, device)
        all_rows.extend(rs)
        if rs:
            avg_lj = np.mean([r["cos_neg_lj"] for r in rs])
            avg_gt = np.mean([r["cos_neg_gt"] for r in rs])
            avg_rand = np.mean([r["cos_random"] for r in rs])
            print(f"  TM={rs[0]['tm']:.3f}  mean_cos: LJ={avg_lj:+.3f}  GT={avg_gt:+.3f}  rand={avg_rand:+.3f}")
        torch.cuda.empty_cache()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {args.out_csv}")

    # summary
    arr_lj   = np.array([r["cos_neg_lj"]    for r in all_rows])
    arr_gt   = np.array([r["cos_neg_gt"]    for r in all_rows])
    arr_fin  = np.array([r["cos_neg_final"] for r in all_rows])
    arr_rnd  = np.array([r["cos_random"]    for r in all_rows])
    print(f"\nGlobal mean cosine ({len(all_rows)} layer-protein samples):")
    print(f"  cos(Δz, -∇LJ)    = {arr_lj.mean():+.4f}   (std {arr_lj.std():.3f})")
    print(f"  cos(Δz, -∇GT)    = {arr_gt.mean():+.4f}   (std {arr_gt.std():.3f})")
    print(f"  cos(Δz, -∇final) = {arr_fin.mean():+.4f}   (std {arr_fin.std():.3f})")
    print(f"  cos(Δz, random)  = {arr_rnd.mean():+.4f}   (std {arr_rnd.std():.3f})")

    # ── plot ──────────────────────────────────────────────────────────────────
    import matplotlib as mpl, matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.12)
    fig.suptitle("Is the trunk doing gradient descent on LJ energy?",
                 x=0.08, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color="#172033")
    fig.text(0.08, 0.87,
             f"cos(Δz_t, −∇L)  averaged across {len(all_rows)} (protein, layer) samples.  "
             "Positive = trunk update aligns with the direction that decreases L.",
             ha="left", fontsize=10, color="#475569")

    layers = sorted({r["layer"] for r in all_rows})
    def per_layer_mean(key):
        m = []
        for L in layers:
            vals = [r[key] for r in all_rows if r["layer"] == L]
            m.append(np.nanmean(vals))
        return np.array(m)

    ax.plot(layers, per_layer_mean("cos_neg_lj"),    color="#dc2626", lw=2.4, label="−∇LJ  (energy)")
    ax.plot(layers, per_layer_mean("cos_neg_gt"),    color="#2563eb", lw=2.4, label="−∇L_GT (truth)")
    ax.plot(layers, per_layer_mean("cos_neg_final"), color="#16a34a", lw=2.4, label="−∇L_final (self-pred)")
    ax.plot(layers, per_layer_mean("cos_random"),    color="#94a3b8", lw=1.6, ls="--", label="random")
    ax.axhline(0, color="#0f172a", lw=0.8)
    ax.set_xlabel("ESMFold trunk layer t", fontsize=12)
    ax.set_ylabel(r"$\cos(\Delta z_t, -\nabla L)$", fontsize=12)
    ax.set_title("Mean alignment of trunk update with negative gradient of each candidate loss", fontsize=11)
    ax.grid(True, lw=0.5)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(args.out_png, dpi=180)
    plt.close(fig)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
