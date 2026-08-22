#!/usr/bin/env python3
"""Attractor-basin steering experiment.

For each similar pair (A, B):
  - Fold sequence A through ESMFold, capture z_20
  - Steer z_20 toward protein B's GT distance map using the cw30 probe
  - Complete the forward pass from layer 20 onward
  - Measure TM(output, A_gt) and TM(output, B_gt)
  - Hypothesis: output moves toward B (accessible neighbouring basin)

For each orphan protein X:
  - Fold X, capture z_20
  - Steer z_20 toward a randomly-selected dissimilar protein's GT distances
  - Measure TM(output, X_gt) and TM(output, random_gt)
  - Hypothesis: output stays near X (isolated attractor, snaps back)

Outputs: results JSON + summary plot
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForProteinFolding

from recurrent_folding.esmfold_probes import ESMFoldTrunkCapture, PairDistogramBottleneck
from recurrent_folding.train_esmfold_distogram_probes import ESMFOLD_DISTO_BINS, distogram_bin_centers


ROOT    = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "runs/general_convergence_ood500_native_cw30_stream"
PROBES  = ROOT / "runs/general_belief_replay_benchmark20_cw30"
INJECT_LAYER = 20


# ── geometry ──────────────────────────────────────────────────────────────────
def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(0); tc = target - target.mean(0)
    h = pc.T @ tc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1., 1., det]) @ u.T).T + target.mean(0)


def tm_score(pred: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(pred), len(ref))
    if n < 5: return float("nan")
    p, r = pred[:n], ref[:n]
    aligned = kabsch_align(p, r)
    d = np.sqrt(((aligned - r) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def pairwise_dist(ca: np.ndarray) -> np.ndarray:
    d = ca[:, None] - ca[None, :]
    return np.sqrt((d * d).sum(-1))


# ── probe steering ────────────────────────────────────────────────────────────
def load_probe(layer: int, device: torch.device) -> PairDistogramBottleneck:
    path = PROBES / f"layer_{layer:02d}" / "probe.pt"
    pr = PairDistogramBottleneck(pair_dim=128, bins=ESMFOLD_DISTO_BINS, hidden_dim=256).to(device).eval()
    pr.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in pr.parameters(): p.requires_grad_(False)
    return pr


def steer_z(z_layer: torch.Tensor, probe: PairDistogramBottleneck,
            target_dist_matrix: np.ndarray, centers: torch.Tensor,
            strength: float, sigma: float, device: torch.device) -> torch.Tensor:
    """Hard-swap the probe's distance prediction to the target for differing pairs.

    For each (i,j), find the distogram bin closest to target_dist_matrix[i,j]
    and replace the probe logits with a sharp peak at that bin.  The residual
    (everything in z the probe cannot represent) is preserved unchanged.
    """
    L = z_layer.shape[0]
    ii, jj = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    ii_f, jj_f = ii.reshape(-1), jj.reshape(-1)

    pairs_flat = z_layer[ii_f, jj_f].to(device)
    ii_d, jj_d = ii_f.to(device), jj_f.to(device)
    c = centers.to(device)  # (n_bins,)

    with torch.no_grad():
        logits = probe.encode(pairs_flat, ii_d, jj_d, L)

        # Residual: the part of z the probe cannot represent
        z_recon_flat = probe.decode(logits, ii_d, jj_d, L)
        residual_flat = pairs_flat - z_recon_flat

        # Hard-assign: replace logits with a peaked distribution at the target bin
        target_flat = torch.tensor(
            target_dist_matrix[ii_f.numpy(), jj_f.numpy()],
            dtype=torch.float32, device=device)          # (N_pairs,)
        target_bins = torch.argmin(torch.abs(c - target_flat[:, None]), dim=1)  # (N_pairs,)
        steered_logits = torch.full_like(logits, -10.0)
        steered_logits.scatter_(1, target_bins.unsqueeze(1), 10.0)

        z_steered_flat = probe.decode(steered_logits, ii_d, jj_d, L)

        # Add the residual back — only the geometry changed
        z_new_flat = z_steered_flat + residual_flat

    z_out = z_new_flat.reshape(L, L, -1).cpu()
    for k in range(L):
        z_out[k, k] = z_layer[k, k]
    return z_out


# ── hook for injecting steered z ──────────────────────────────────────────────
class InjectHook:
    def __init__(self, z_steered: torch.Tensor):
        self.z = z_steered

    def __call__(self, _mod, _inp, out):
        s, z = out
        return (s, self.z[None].to(device=z.device, dtype=z.dtype))


# ── single protein: fold + optionally steer ───────────────────────────────────
def fold(model, tokenizer, capture, seq: str,
         inject_layer: int | None = None,
         z_steered: torch.Tensor | None = None,
         device: torch.device = torch.device("cpu")) -> np.ndarray:
    ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    handle = None
    if inject_layer is not None and z_steered is not None:
        hook = InjectHook(z_steered)
        handle = model.trunk.blocks[inject_layer].register_forward_hook(hook)
    ctx = capture.capturing() if inject_layer is None else torch.no_grad()
    with ctx, torch.no_grad():
        out = model(ids)
    if handle is not None:
        handle.remove()
    return out["positions"][-1, 0, :, 1, :].detach().cpu().float().numpy()


# ── load GT CA for a protein ──────────────────────────────────────────────────
def load_gt(name: str, ood_dir: Path) -> tuple[np.ndarray, str]:
    d = ood_dir / f"{name}_analysis"
    gt_p = d / "run_artifacts/gt_ca_distances.npz"
    src_p = d / "source.json"
    gt = np.load(gt_p)
    ca = gt["ca_coords"].astype(float)
    seq_raw = gt.get("sequence", None)
    if seq_raw is not None:
        seq = str(seq_raw)
    else:
        src = json.loads(src_p.read_text())
        seq = src.get("sequence", "")
    return ca, seq


# ── run one steering trial ────────────────────────────────────────────────────
def run_trial(model, tokenizer, capture, probe, centers,
              src_name: str, src_seq: str, src_gt_ca: np.ndarray,
              tgt_name: str, tgt_gt_ca: np.ndarray,
              strength: float, sigma: float, device: torch.device) -> dict:
    # 1. baseline fold of source
    t0 = time.time()
    ca_base = fold(model, tokenizer, capture, src_seq, device=device)

    # 2. capture z at inject layer
    z_full = capture.stacked_z().squeeze(1)  # (48, L, L, 128)
    z_layer = z_full[INJECT_LAYER].float()

    # 3. target distance matrix (GT of target protein, cropped to source length)
    L_src = len(src_seq)
    L_tgt = len(tgt_gt_ca)
    L_use = min(L_src, L_tgt)
    tgt_dist = pairwise_dist(tgt_gt_ca[:L_use])
    # pad to L_src if needed
    if L_use < L_src:
        pad = np.full((L_src, L_src), tgt_dist.mean())
        pad[:L_use, :L_use] = tgt_dist
        tgt_dist = pad

    # 4. steer
    z_s = steer_z(z_layer, probe, tgt_dist, centers, strength, sigma, device)

    # 5. steered forward pass
    ca_steer = fold(model, tokenizer, capture, src_seq,
                    inject_layer=INJECT_LAYER, z_steered=z_s, device=device)

    elapsed = time.time() - t0

    # 6. pairwise distance MAE (upper triangle only)
    n = min(len(ca_base), len(ca_steer), len(src_gt_ca), len(tgt_gt_ca))
    iu = np.triu_indices(n, k=1)

    D_base  = pairwise_dist(ca_base[:n])[iu]
    D_steer = pairwise_dist(ca_steer[:n])[iu]
    D_src   = pairwise_dist(src_gt_ca[:n])[iu]
    D_tgt   = pairwise_dist(tgt_gt_ca[:n])[iu]

    base_mae_src  = float(np.abs(D_base  - D_src).mean())
    steer_mae_src = float(np.abs(D_steer - D_src).mean())
    base_mae_tgt  = float(np.abs(D_base  - D_tgt).mean())
    steer_mae_tgt = float(np.abs(D_steer - D_tgt).mean())

    ca_rmsd = float(np.sqrt(np.mean((ca_base[:n] - ca_steer[:n]) ** 2)))

    # 7. Directional accuracy on high-disagreement pairs
    #    Restrict to pairs where |D_src - D_tgt| is in the top 25 %
    #    (the pairs where the steer actually does meaningful work).
    #    For each such pair: did D_steer move in the right direction toward D_tgt?
    disagreement = np.abs(D_tgt - D_src)
    thresh25 = float(np.percentile(disagreement, 75))
    mask = disagreement >= thresh25          # top-quartile disagreement pairs
    n_disagree = int(mask.sum())
    if n_disagree > 0:
        needed = np.sign(D_tgt[mask] - D_base[mask])   # direction we want
        actual = np.sign(D_steer[mask] - D_base[mask]) # direction we got
        non_tie = needed != 0                           # skip pairs already at target
        if non_tie.sum() > 0:
            dir_acc = float((needed[non_tie] == actual[non_tie]).mean())
        else:
            dir_acc = float("nan")
        # mean signed movement in the correct direction (positive = moved right)
        move_toward_tgt = float(((D_steer[mask] - D_base[mask]) * needed).mean())
    else:
        dir_acc = float("nan")
        move_toward_tgt = float("nan")

    return {
        "src": src_name, "tgt": tgt_name,
        "L_src": L_src, "L_tgt": L_tgt,
        "base_mae_vs_src":       round(base_mae_src,  3),
        "steer_mae_vs_src":      round(steer_mae_src, 3),
        "base_mae_vs_tgt":       round(base_mae_tgt,  3),
        "steer_mae_vs_tgt":      round(steer_mae_tgt, 3),
        "delta_mae_tgt":         round(steer_mae_tgt - base_mae_tgt, 3),
        "dir_acc_top25pct":      round(dir_acc, 4) if not np.isnan(dir_acc) else None,
        "move_toward_tgt_top25": round(move_toward_tgt, 4) if not np.isnan(move_toward_tgt) else None,
        "n_disagree_pairs":      n_disagree,
        "disagree_thresh_ang":   round(thresh25, 3),
        "ca_rmsd_base_vs_steer": round(ca_rmsd, 3),
        "elapsed_s": round(elapsed, 1),
    }


# ── plot ──────────────────────────────────────────────────────────────────────
def plot_results(pairs_results: list[dict], orphan_results: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.13, wspace=0.38)
    fig.suptitle("Attractor-basin steering: probe faithfulness via distance steering",
                 x=0.06, y=0.96, ha="left", fontsize=17, fontweight="bold", color="#172033")
    fig.text(0.06, 0.905,
             f"Layer {INJECT_LAYER} z steered toward target GT distance map via cw30 probe (residual preserved).  "
             "Directional accuracy = fraction of high-disagreement pairs (top 25% |D_A−D_B|) "
             "where steered distance moved toward D_B.",
             ha="left", fontsize=11, color="#475569")

    all_results = pairs_results + orphan_results
    pair_labels  = [f"{r['src']}→{r['tgt']}" for r in pairs_results]
    orphan_labels = [r["src"] for r in orphan_results]

    # Panel 0: directional accuracy (top-25% disagreement pairs)
    ax = axes[0]
    p_da = [r["dir_acc_top25pct"] for r in pairs_results]
    o_da = [r["dir_acc_top25pct"] for r in orphan_results]
    xs_p = np.zeros(len(p_da)); xs_o = np.ones(len(o_da))
    ax.scatter(xs_p, p_da, s=110, color="#16a34a", edgecolors="#172033",
               lw=0.7, zorder=3, label=f"similar pairs (n={len(p_da)})")
    ax.scatter(xs_o, o_da, s=110, color="#ef4444", edgecolors="#172033",
               lw=0.7, zorder=3, label=f"orphans (n={len(o_da)})")
    ax.axhline(0.5, color="#94a3b8", lw=1.4, ls="--", label="random (0.50)")
    for lbl, x, y in zip(pair_labels, xs_p, p_da):
        if y is not None:
            ax.annotate(lbl, (x, y), fontsize=6, xytext=(6, 0), textcoords="offset points")
    for lbl, x, y in zip(orphan_labels, xs_o, o_da):
        if y is not None:
            ax.annotate(lbl, (x, y), fontsize=6, xytext=(6, 0), textcoords="offset points")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Similar pairs", "Orphans"] if orphan_results else ["Similar pairs"], fontsize=11)
    ax.set_ylabel("Directional accuracy (top-25% disagreement pairs)", fontsize=10)
    ax.set_title("Probe faithfulness: did distances move\ntoward target?", fontsize=11)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, lw=0.5)
    if p_da:
        valid = [v for v in p_da if v is not None]
        ax.text(0.04, 0.04, f"mean={np.mean(valid):.3f}", transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#dcfce7", ec="#15803d", alpha=0.9))

    # Panel 1: ΔMAE toward target per pair
    ax2 = axes[1]
    p_dm = [r["delta_mae_tgt"] for r in pairs_results]
    colors = ["#16a34a" if d < 0 else "#ef4444" for d in p_dm]
    bars = ax2.barh(range(len(p_dm)), p_dm, color=colors, edgecolor="#172033", lw=0.6)
    ax2.axvline(0, color="#0f172a", lw=1.2)
    ax2.set_yticks(range(len(p_dm)))
    ax2.set_yticklabels(pair_labels, fontsize=8)
    ax2.set_xlabel("ΔMAE→target (steered − baseline, Å)\nnegative = got closer", fontsize=9)
    ax2.set_title("ΔMAE toward target\n(lower = more toward B)", fontsize=11)
    ax2.grid(True, lw=0.5, axis="x")

    # Panel 2: move_toward_tgt (mean signed movement on top-25% pairs)
    ax3 = axes[2]
    p_mv = [r["move_toward_tgt_top25"] for r in pairs_results]
    colors3 = ["#16a34a" if (v is not None and v > 0) else "#ef4444" for v in p_mv]
    bars3 = ax3.barh(range(len(p_mv)),
                     [v if v is not None else 0.0 for v in p_mv],
                     color=colors3, edgecolor="#172033", lw=0.6)
    ax3.axvline(0, color="#0f172a", lw=1.2)
    ax3.set_yticks(range(len(p_mv)))
    ax3.set_yticklabels(pair_labels, fontsize=8)
    ax3.set_xlabel("Mean signed movement toward D_tgt (Å)\npositive = moved in right direction", fontsize=9)
    ax3.set_title("Mean movement toward target\n(top-25% disagreement pairs)", fontsize=11)
    ax3.grid(True, lw=0.5, axis="x")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json",   type=Path,
                    default=OOD_DIR / "attractor_pairs.json")
    ap.add_argument("--ood-dir",      type=Path, default=OOD_DIR)
    ap.add_argument("--probes-root",  type=Path, default=PROBES)
    ap.add_argument("--out-dir",      type=Path,
                    default=ROOT / "runs/attractor_experiment_mae")
    ap.add_argument("--inject-layer", type=int,  default=INJECT_LAYER)
    ap.add_argument("--strength",     type=float, default=6.0)
    ap.add_argument("--sigma",        type=float, default=2.0)
    ap.add_argument("--device",       default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs_data = json.loads(args.pairs_json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ESMFold...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
    if device.type == "cuda":
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)

    capture = ESMFoldTrunkCapture(model)
    probe   = load_probe(args.inject_layer, device)
    centers = distogram_bin_centers(device, torch.float32)

    # ── similar pairs ────────────────────────────────────────────────────────
    print(f"\n=== Similar pairs (n={len(pairs_data['pairs'])}) ===")
    pairs_results = []
    for entry in pairs_data["pairs"]:
        na, nb = entry["name_a"], entry["name_b"]
        print(f"  {na} → {nb}  (TM_pred={entry['tm_pred']:.3f})")
        src_ca, src_seq = load_gt(na, args.ood_dir)
        tgt_ca, _       = load_gt(nb, args.ood_dir)
        r = run_trial(model, tokenizer, capture, probe, centers,
                      na, src_seq, src_ca, nb, tgt_ca,
                      args.strength, args.sigma, device)
        pairs_results.append(r)
        dir_s = f"{r['dir_acc_top25pct']:.3f}" if r['dir_acc_top25pct'] is not None else "nan"
        mv_s  = f"{r['move_toward_tgt_top25']:+.3f}" if r['move_toward_tgt_top25'] is not None else "nan"
        print(f"    ΔMAE→tgt={r['delta_mae_tgt']:+.2f}  "
              f"dir_acc(top25%)={dir_s}  move_toward={mv_s}  "
              f"n_disagree={r['n_disagree_pairs']}  rmsd={r['ca_rmsd_base_vs_steer']:.2f}Å")

    # ── orphans ──────────────────────────────────────────────────────────────
    print(f"\n=== Orphan proteins (n={len(pairs_data['orphans'])}) ===")
    # Build pool of dissimilar proteins to use as random steer targets
    all_names = [r["name_a"] for r in pairs_data["pairs"]] + \
                [r["name_b"] for r in pairs_data["pairs"]]
    rng = np.random.default_rng(42)
    orphan_results = []
    for i, entry in enumerate(pairs_data["orphans"]):
        nx = entry["name"]
        # pick random target from similar-pair proteins (guaranteed different fold)
        tgt_name = all_names[int(rng.integers(0, len(all_names)))]
        while tgt_name == nx:
            tgt_name = all_names[int(rng.integers(0, len(all_names)))]
        print(f"  {nx} → {tgt_name} (random steer, max_TM={entry['max_tm_to_others']:.3f})")
        src_ca, src_seq = load_gt(nx, args.ood_dir)
        tgt_ca, _       = load_gt(tgt_name, args.ood_dir)
        r = run_trial(model, tokenizer, capture, probe, centers,
                      nx, src_seq, src_ca, tgt_name, tgt_ca,
                      args.strength, args.sigma, device)
        orphan_results.append(r)
        dir_s = f"{r['dir_acc_top25pct']:.3f}" if r['dir_acc_top25pct'] is not None else "nan"
        mv_s  = f"{r['move_toward_tgt_top25']:+.3f}" if r['move_toward_tgt_top25'] is not None else "nan"
        print(f"    ΔMAE→tgt={r['delta_mae_tgt']:+.2f}  "
              f"dir_acc(top25%)={dir_s}  move_toward={mv_s}  "
              f"n_disagree={r['n_disagree_pairs']}  rmsd={r['ca_rmsd_base_vs_steer']:.2f}Å")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    p_dm  = [r["delta_mae_tgt"] for r in pairs_results]
    o_dm  = [r["delta_mae_tgt"] for r in orphan_results]
    p_da  = [r["dir_acc_top25pct"]      for r in pairs_results if r["dir_acc_top25pct"]      is not None]
    o_da  = [r["dir_acc_top25pct"]      for r in orphan_results if r["dir_acc_top25pct"]     is not None]
    p_mv  = [r["move_toward_tgt_top25"] for r in pairs_results if r["move_toward_tgt_top25"] is not None]
    o_mv  = [r["move_toward_tgt_top25"] for r in orphan_results if r["move_toward_tgt_top25"] is not None]

    print(f"Similar pairs  ΔMAE→tgt  mean={np.mean(p_dm):+.3f}  "
          f"improved(ΔMAE<0)={sum(d<0 for d in p_dm)}/{len(p_dm)}")
    if p_da:
        print(f"               dir_acc(top25%)  mean={np.mean(p_da):.3f}  "
              f">0.5: {sum(d>0.5 for d in p_da)}/{len(p_da)}")
        print(f"               move_toward(top25%) mean={np.mean(p_mv):+.3f}")
    if orphan_results:
        print(f"Orphans        ΔMAE→tgt  mean={np.mean(o_dm):+.3f}  "
              f"improved(ΔMAE<0)={sum(d<0 for d in o_dm)}/{len(o_dm)}")
        if o_da:
            print(f"               dir_acc(top25%)  mean={np.mean(o_da):.3f}  "
                  f">0.5: {sum(d>0.5 for d in o_da)}/{len(o_da)}")

    results = {"pairs": pairs_results, "orphans": orphan_results,
               "inject_layer": args.inject_layer,
               "strength": args.strength, "sigma": args.sigma}
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    plot_results(pairs_results, orphan_results,
                 args.out_dir / "attractor_experiment.png")


if __name__ == "__main__":
    main()
