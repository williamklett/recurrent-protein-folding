#!/usr/bin/env python3
"""Clean 2x3 figure: pLDDT alone vs pLDDT+convexity, across natural + synthetic.

Top row: natural benchmark (n=20, target = log Cα RMSD)
Bot row: synthetic OOD benchmark (n=29, target = TM-score)

Cols:
  1) pLDDT alone
  2) pLDDT + log_late_amp
  3) pLDDT + TTC98
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


def convexity_metrics(corr):
    eps = 1e-9
    c = corr.astype(float); L = len(c)
    start = min(17, L // 3); c_late = c[start:]
    diffs = np.diff(c_late)
    neg_step = float(-diffs[diffs < 0].sum()) if (diffs < 0).any() else 0.0
    late_amp = float(c[35:].max() - c[35:].min()) if L > 36 else float("nan")
    auc_late = float((1.0 - c[35:]).sum()) if L > 36 else float("nan")
    above98 = np.where(c >= 0.98)[0]
    ttc98 = float(above98[0] / L) if len(above98) > 0 else 1.0
    return {
        "log_late_amp": float(np.log10(max(late_amp, eps))),
        "log_neg_step": float(np.log10(max(neg_step, eps))),
        "log_auc_late": float(np.log10(max(auc_late, eps))),
        "ttc98": ttc98,
    }


def kabsch_align(p, q):
    pc = p - p.mean(0); qc = q - q.mean(0)
    h = pc.T @ qc; u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    return pc @ (vt.T @ np.diag([1,1,det]) @ u.T).T + q.mean(0)


def tm_for(d):
    p = d / "esmfold_direct_eval/esmfold_prediction_ca.npz"
    g = d / "run_artifacts/gt_ca_distances.npz"
    if not p.exists() or not g.exists(): return float("nan")
    pa = np.load(p)["pred_ca"].astype(float)
    gt = np.load(g)["ca_coords"].astype(float)
    n = min(len(pa), len(gt))
    if n < 3: return float("nan")
    pa = pa[:n]; gt = gt[:n]
    aligned = kabsch_align(pa, gt)
    dists = np.sqrt(((aligned - gt) ** 2).sum(-1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))


def load_natural(root):
    rows = []
    for d in sorted(root.glob("*_analysis")):
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists(): continue
        m = json.loads(metrics_p.read_text())
        corr = np.array([float(r["corr_vs_47"]) for r in csv.DictReader(corr_p.open())])
        cm = convexity_metrics(corr)
        rows.append({
            "name": d.name.removesuffix("_analysis"),
            "rmsd": float(m["ca_rmsd_after_kabsch"]),
            "plddt": float(m["mean_plddt"]),
            **cm,
        })
    return rows


def load_synthetic(correct_root, miss_root):
    rows = []
    for root, group in [(correct_root, "correct"), (miss_root, "miss")]:
        for d in sorted(root.glob("*_analysis")):
            metrics_p = d / "esmfold_direct_eval/metrics.json"
            corr_p = d / "layerwise_distance_correlations.csv"
            if not metrics_p.exists() or not corr_p.exists(): continue
            m = json.loads(metrics_p.read_text())
            corr = np.array([float(r["corr_vs_47"]) for r in csv.DictReader(corr_p.open())])
            cm = convexity_metrics(corr)
            tm = tm_for(d)
            rows.append({
                "name": d.name.removesuffix("_analysis"),
                "group": group,
                "tm": tm, "plddt": float(m["mean_plddt"]),
                **cm,
            })
    # dedupe by name
    seen, dr = set(), []
    for r in rows:
        if r["name"] in seen: continue
        seen.add(r["name"]); dr.append(r)
    return [r for r in dr if not np.isnan(r["tm"])]


def pearson(x, y):
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_lin(X, y):
    X_c = X - X.mean(0); y_c = y - y.mean()
    coef, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
    return X_c @ coef + y.mean()


def panel(ax, x, y, colors, names, x_label, y_label, title, r_val,
          y_threshold=None, threshold_text=None, x_thresh_signs=None):
    ax.scatter(x, y, c=colors, s=120, edgecolors="#172033", lw=0.6)
    for nm, xv, yv in zip(names, x, y):
        ax.annotate(nm, (xv, yv), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if y_threshold is not None:
        ax.axhline(y_threshold, color="#0f172a", lw=0.7, ls="--", alpha=0.45)
        if threshold_text:
            ax.text(0.99, y_threshold, threshold_text, ha="right",
                    va="bottom", fontsize=8, color="#334155",
                    transform=ax.get_yaxis_transform())
    ax.text(0.04, 0.96, f"|r| = {abs(r_val):.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#a16207", alpha=0.95))
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12)
    ax.grid(True, lw=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natural-root", type=Path, required=True)
    ap.add_argument("--synth-correct-root", type=Path, required=True)
    ap.add_argument("--synth-miss-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    style()

    nat = load_natural(args.natural_root)
    syn = load_synthetic(args.synth_correct_root, args.synth_miss_root)
    print(f"natural n={len(nat)}, synthetic n={len(syn)}")

    # ── Natural data: target = −log₁₀(RMSD)  so higher = better ─────────────
    nat_rmsd = np.array([r["rmsd"] for r in nat])
    nat_y = -np.log10(np.maximum(nat_rmsd, 1e-3))   # higher = better
    nat_plddt = np.array([r["plddt"] for r in nat])
    nat_lateamp = np.array([r["log_late_amp"] for r in nat])
    nat_ttc98 = np.array([r["ttc98"] for r in nat])
    nat_names = [r["name"] for r in nat]
    nat_colors = ["#2f7d5a" if r < 5 else ("#ef6c00" if r < 15 else "#c62828") for r in nat_rmsd]

    nat_r_plddt = pearson(nat_plddt, nat_y)
    nat_combo_2 = fit_lin(np.column_stack([nat_plddt, nat_lateamp]), nat_y)
    nat_r_2 = pearson(nat_combo_2, nat_y)
    nat_combo_3 = fit_lin(np.column_stack([nat_plddt, nat_lateamp, nat_ttc98]), nat_y)
    nat_r_3 = pearson(nat_combo_3, nat_y)

    # ── Synthetic data ───────────────────────────────────────────────────────
    syn_tm = np.array([r["tm"] for r in syn])
    syn_plddt = np.array([r["plddt"] for r in syn])
    syn_lateamp = np.array([r["log_late_amp"] for r in syn])
    syn_ttc98 = np.array([r["ttc98"] for r in syn])
    syn_names = [r["name"] for r in syn]
    syn_colors = ["#2f7d5a" if r["group"] == "correct" else "#d74b2f" for r in syn]

    syn_r_plddt = pearson(syn_plddt, syn_tm)
    syn_combo_2 = fit_lin(np.column_stack([syn_plddt, syn_lateamp]), syn_tm)
    syn_r_2 = pearson(syn_combo_2, syn_tm)
    syn_combo_3 = fit_lin(np.column_stack([syn_plddt, syn_lateamp, syn_ttc98]), syn_tm)
    syn_r_3 = pearson(syn_combo_3, syn_tm)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.subplots_adjust(left=0.055, right=0.98, top=0.90, bottom=0.06, wspace=0.27, hspace=0.32)
    fig.suptitle("Convexity metrics add information to pLDDT — accuracy correlation improves in both domains",
                 x=0.055, y=0.955, ha="left", fontsize=20, fontweight="bold", color="#172033")
    fig.text(0.055, 0.92,
             "Top: natural benchmark (n=20, target = −log Cα RMSD, higher = better).   "
             "Bottom: synthetic OOD benchmark (n=29, target = TM-score, higher = better).",
             ha="left", fontsize=12, color="#475569")

    # Top row: NATURAL
    panel(axes[0, 0], nat_plddt, nat_y, nat_colors, nat_names,
          "ESMFold mean pLDDT", "−log₁₀(Cα RMSD)",
          "Natural — pLDDT alone", nat_r_plddt,
          y_threshold=-np.log10(5), threshold_text="5 Å threshold")
    panel(axes[0, 1], nat_combo_2, nat_y, nat_colors, nat_names,
          "pLDDT + log_late_amp", "−log₁₀(Cα RMSD)",
          "Natural — pLDDT + log_late_amp", nat_r_2,
          y_threshold=-np.log10(5), threshold_text="5 Å threshold")
    panel(axes[0, 2], nat_combo_3, nat_y, nat_colors, nat_names,
          "pLDDT + log_late_amp + TTC98", "−log₁₀(Cα RMSD)",
          "Natural — pLDDT + log_late_amp + TTC98", nat_r_3,
          y_threshold=-np.log10(5), threshold_text="5 Å threshold")

    # Bottom row: SYNTHETIC
    panel(axes[1, 0], syn_plddt, syn_tm, syn_colors, syn_names,
          "ESMFold mean pLDDT", "TM-score vs PDB",
          "Synthetic OOD — pLDDT alone", syn_r_plddt,
          y_threshold=0.5, threshold_text="TM=0.5 fold threshold")
    panel(axes[1, 1], syn_combo_2, syn_tm, syn_colors, syn_names,
          "pLDDT + log_late_amp", "TM-score vs PDB",
          "Synthetic OOD — pLDDT + log_late_amp", syn_r_2,
          y_threshold=0.5, threshold_text="TM=0.5 fold threshold")
    panel(axes[1, 2], syn_combo_3, syn_tm, syn_colors, syn_names,
          "pLDDT + log_late_amp + TTC98", "TM-score vs PDB",
          "Synthetic OOD — pLDDT + log_late_amp + TTC98", syn_r_3,
          y_threshold=0.5, threshold_text="TM=0.5 fold threshold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200); plt.close(fig)
    print(f"wrote {args.out}")

    # Print summary
    print(f"\nNatural   (n={len(nat)}, target = −log RMSD, higher = better):")
    print(f"  pLDDT alone                           : |r| = {abs(nat_r_plddt):.3f}")
    print(f"  pLDDT + log_late_amp                  : |r| = {abs(nat_r_2):.3f}   Δ = {abs(nat_r_2)-abs(nat_r_plddt):+.3f}")
    print(f"  pLDDT + log_late_amp + TTC98          : |r| = {abs(nat_r_3):.3f}   Δ = {abs(nat_r_3)-abs(nat_r_plddt):+.3f}")
    print(f"\nSynthetic (n={len(syn)}, target = TM, higher = better):")
    print(f"  pLDDT alone                           : |r| = {abs(syn_r_plddt):.3f}")
    print(f"  pLDDT + log_late_amp                  : |r| = {abs(syn_r_2):.3f}   Δ = {abs(syn_r_2)-abs(syn_r_plddt):+.3f}")
    print(f"  pLDDT + log_late_amp + TTC98          : |r| = {abs(syn_r_3):.3f}   Δ = {abs(syn_r_3)-abs(syn_r_plddt):+.3f}")


if __name__ == "__main__":
    main()
