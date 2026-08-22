#!/usr/bin/env python3
"""Test candidate convexity metrics on the synthetic-design benchmark.

For each protein's corr_vs_47 curve, compute several non-convexity scores
(0 = perfectly convex/monotonic, larger = more non-convex). Compare each to
TM-score and to ESMFold's pLDDT to see which best separates correct designs
from misses.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def kabsch_align(pred, target):
    pc = pred - pred.mean(axis=0); tc = target - target.mean(axis=0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, det]) @ u.T
    return pc @ rot.T + target.mean(axis=0)


def tm_score(d_analysis: Path) -> float:
    pred_p = d_analysis / "esmfold_direct_eval/esmfold_prediction_ca.npz"
    gt_p = d_analysis / "run_artifacts/gt_ca_distances.npz"
    if not pred_p.exists() or not gt_p.exists():
        return float("nan")
    pred = np.load(pred_p)["pred_ca"].astype(float)
    gt = np.load(gt_p)["ca_coords"].astype(float)
    n = min(len(pred), len(gt))
    if n < 3: return float("nan")
    pred = pred[:n]; gt = gt[:n]
    pa = kabsch_align(pred, gt)
    distances = np.sqrt(((pa - gt) ** 2).sum(axis=1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def convexity_metrics(corr: np.ndarray) -> dict[str, float]:
    """Compute non-convexity scores. Larger = more non-convex."""
    eps = 1e-9
    c = corr.astype(float)
    L = len(c)
    start = min(17, L // 3)
    c_late = c[start:]

    diffs = np.diff(c_late)
    neg_step = float(-diffs[diffs < 0].sum()) if (diffs < 0).any() else 0.0
    run_max = np.maximum.accumulate(c_late)
    drawdown_curve = run_max - c_late
    max_drawdown = float(drawdown_curve.max())
    deficit_area = float(drawdown_curve.sum())
    n_local_min = sum(1 for i in range(1, len(c_late) - 1)
                      if c_late[i] < c_late[i - 1] and c_late[i] < c_late[i + 1])
    d2 = np.diff(c_late, n=2)
    d2_std = float(d2.std()) if len(d2) > 0 else 0.0
    late_amp = float(c[35:].max() - c[35:].min()) if L > 36 else float("nan")

    # AUC deficit: area between curve and 1.0 over the WHOLE trunk
    auc_deficit = float((1.0 - c).sum())
    # Late-AUC deficit: area between curve and 1.0 over blocks 36+
    auc_late = float((1.0 - c[35:]).sum()) if L > 36 else float("nan")

    # Layer at which corr first exceeds 0.99 (normalized to [0,1])
    above = np.where(c >= 0.99)[0]
    ttc99 = float(above[0] / L) if len(above) > 0 else 1.0
    above98 = np.where(c >= 0.98)[0]
    ttc98 = float(above98[0] / L) if len(above98) > 0 else 1.0

    # Step jump in last 5 blocks
    if L >= 5:
        last5_step = float(c[-1] - c[-5])
    else:
        last5_step = 0.0

    # Log-transformed versions of the convexity scores (compress dynamic range)
    log_neg_step     = float(np.log10(max(neg_step, eps)))
    log_max_drawdown = float(np.log10(max(max_drawdown, eps)))
    log_deficit_area = float(np.log10(max(deficit_area, eps)))
    log_d2_std       = float(np.log10(max(d2_std, eps)))
    log_late_amp     = float(np.log10(max(late_amp if not np.isnan(late_amp) else eps, eps)))
    log_auc_deficit  = float(np.log10(max(auc_deficit, eps)))
    log_auc_late     = float(np.log10(max(auc_late if not np.isnan(auc_late) else eps, eps)))

    return {
        "neg_step": neg_step, "max_drawdown": max_drawdown,
        "deficit_area": deficit_area, "n_local_min": float(n_local_min),
        "d2_std": d2_std, "late_amp": late_amp,
        "auc_deficit": auc_deficit, "auc_late": auc_late,
        "ttc99": ttc99, "ttc98": ttc98, "last5_step": last5_step,
        "log_neg_step": log_neg_step, "log_max_drawdown": log_max_drawdown,
        "log_deficit_area": log_deficit_area, "log_d2_std": log_d2_std,
        "log_late_amp": log_late_amp,
        "log_auc_deficit": log_auc_deficit, "log_auc_late": log_auc_late,
    }


def load_rows(root: Path, group: str) -> list[dict]:
    out = []
    for d in sorted(root.glob("*_analysis")):
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p   = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists(): continue
        m = json.loads(metrics_p.read_text())
        corr_rows = list(csv.DictReader(corr_p.open()))
        layers = np.array([int(r["layer"]) for r in corr_rows])
        corr = np.array([float(r["corr_vs_47"]) for r in corr_rows])
        cm = convexity_metrics(corr)
        out.append({
            "name": d.name.removesuffix("_analysis"),
            "group": group,
            "plddt": float(m["mean_plddt"]),
            "mae": float(m["distance_mae_all_pairs"]),
            "tm": tm_score(d),
            **cm,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct-root", type=Path, required=True)
    ap.add_argument("--miss-root", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    style()

    rows = load_rows(args.correct_root, "correct") + load_rows(args.miss_root, "miss")
    # dedupe by name
    seen = set(); dr = []
    for r in rows:
        if r["name"] in seen: continue
        seen.add(r["name"]); dr.append(r)
    rows = dr

    metrics_to_test = [
        "plddt",
        "late_amp", "log_late_amp",
        "neg_step", "log_neg_step",
        "max_drawdown", "log_max_drawdown",
        "deficit_area", "log_deficit_area",
        "n_local_min", "d2_std", "log_d2_std",
        "auc_deficit", "log_auc_deficit",
        "auc_late", "log_auc_late",
        "ttc99", "ttc98", "last5_step",
    ]
    print(f"\n N={len(rows)}\n")
    print(f"{'metric':>15} {'r_vs_TM':>10} {'r_vs_MAE':>10}")
    print("-" * 40)

    tm = np.array([r["tm"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    valid = ~np.isnan(tm)
    perf = {}
    for m in metrics_to_test:
        x = np.array([r[m] for r in rows])
        # pLDDT is positive (higher = better); others are non-convexity (higher = worse)
        sign_TM  = 1 if m == "plddt" else -1
        sign_MAE = -1 if m == "plddt" else 1
        r_tm  = float(np.corrcoef(x[valid], tm[valid])[0,1]) if valid.sum() > 2 else float("nan")
        r_mae = float(np.corrcoef(x, mae)[0,1])
        perf[m] = (r_tm, r_mae)
        print(f"{m:>15} {r_tm:>+10.3f} {r_mae:>+10.3f}")

    # Rank metrics by |r_vs_TM|
    ranked = sorted(perf.items(), key=lambda kv: -abs(kv[1][0]) if not np.isnan(kv[1][0]) else 0)
    print("\n=== ranked by |r| with TM-score ===")
    for m, (r_tm, r_mae) in ranked:
        print(f"{m:>20} {r_tm:>+8.3f}")

    # Plot top 8 vs TM-score
    top_metrics = [m for m, _ in ranked[:8]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Non-convexity metrics vs TM-score", x=0.05, y=0.97,
                 ha="left", fontsize=22, fontweight="bold", color="#172033")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.91, bottom=0.07,
                        wspace=0.27, hspace=0.32)

    for ax, m in zip(axes.flat, top_metrics):
        for group, color, marker in [("correct", "#2f7d5a", "o"), ("miss", "#d74b2f", "^")]:
            xs = [r[m] for r in rows if r["group"] == group]
            ys = [r["tm"] for r in rows if r["group"] == group]
            ax.scatter(xs, ys, c=color, marker=marker, s=120, edgecolors="#172033", lw=0.6,
                       label=f"{group} (n={len(xs)})")
            for r in [r for r in rows if r["group"] == group]:
                ax.annotate(r["name"], (r[m], r["tm"]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
        ax.axhline(0.5, color="#0f172a", lw=0.8, ls="--", alpha=0.5)
        r_tm = perf[m][0]
        ax.text(0.04, 0.96, f"r = {r_tm:+.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)
        ax.set_xlabel(m); ax.set_ylabel("TM-score vs PDB"); ax.grid(True, lw=0.6)
        ax.set_title(m, fontsize=12)
    axes.flat[0].legend(loc="lower right", fontsize=8)
    # Hide unused panel
    axes.flat[-1].set_visible(False)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    p = args.out_prefix.with_name(args.out_prefix.name + "_convexity_panels.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"\nwrote {p}")

    # Write CSV
    p_csv = args.out_prefix.with_name(args.out_prefix.name + "_convexity.csv")
    fields = ["group", "name", "tm", "mae", "plddt"] + [m for m in metrics_to_test if m != "plddt"]
    with p_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["group"], x["mae"])):
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {p_csv}")


if __name__ == "__main__":
    main()
