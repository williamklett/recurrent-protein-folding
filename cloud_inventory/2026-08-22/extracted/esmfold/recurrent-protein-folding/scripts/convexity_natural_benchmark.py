#!/usr/bin/env python3
"""Run convexity metrics on the 20-protein natural benchmark.
Correlates with Cα RMSD (and distance MAE) rather than TM-score, since the
natural benchmark uses those as the canonical accuracy metrics."""
from __future__ import annotations
import argparse, csv, itertools, json
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


def convexity_metrics(corr: np.ndarray) -> dict[str, float]:
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
    d2 = np.diff(c_late, n=2)
    d2_std = float(d2.std()) if len(d2) > 0 else 0.0
    late_amp = float(c[35:].max() - c[35:].min()) if L > 36 else float("nan")
    auc_late = float((1.0 - c[35:]).sum()) if L > 36 else float("nan")
    above98 = np.where(c >= 0.98)[0]
    ttc98 = float(above98[0] / L) if len(above98) > 0 else 1.0
    above99 = np.where(c >= 0.99)[0]
    ttc99 = float(above99[0] / L) if len(above99) > 0 else 1.0
    return {
        "ttc98": ttc98, "ttc99": ttc99,
        "late_amp": late_amp, "log_late_amp": float(np.log10(max(late_amp, eps))),
        "neg_step": neg_step, "log_neg_step": float(np.log10(max(neg_step, eps))),
        "max_drawdown": max_drawdown,
        "log_max_drawdown": float(np.log10(max(max_drawdown, eps))),
        "deficit_area": deficit_area,
        "log_deficit_area": float(np.log10(max(deficit_area, eps))),
        "auc_late": auc_late,
        "log_auc_late": float(np.log10(max(auc_late, eps))),
        "d2_std": d2_std, "log_d2_std": float(np.log10(max(d2_std, eps))),
    }


def load_rows(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("*_analysis")):
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists(): continue
        m = json.loads(metrics_p.read_text())
        corr_rows = list(csv.DictReader(corr_p.open()))
        corr = np.array([float(r["corr_vs_47"]) for r in corr_rows])
        cm = convexity_metrics(corr)
        out.append({
            "name": d.name.removesuffix("_analysis"),
            "rmsd": float(m["ca_rmsd_after_kabsch"]),
            "mae": float(m["distance_mae_all_pairs"]),
            "dist_corr": float(m["distance_corr_all_pairs"]),
            "plddt": float(m["mean_plddt"]),
            **cm,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    style()

    rows = load_rows(args.root)
    n = len(rows)
    print(f"N={n}")

    def get(m): return np.array([r[m] for r in rows])

    def pearson(x, y):
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12: return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    rmsd = get("rmsd")
    log_rmsd = np.log10(np.maximum(rmsd, 1e-3))

    candidate_metrics = [
        "plddt",
        "ttc98", "ttc99",
        "log_late_amp", "log_neg_step", "log_auc_late",
        "log_max_drawdown", "log_deficit_area", "log_d2_std",
    ]
    print(f"\n{'metric':>20}  {'r_vs_RMSD':>10}  {'r_vs_log(RMSD)':>16}")
    for m in candidate_metrics:
        x = get(m)
        print(f"{m:>20}  {pearson(x, rmsd):>+10.3f}  {pearson(x, log_rmsd):>+16.3f}")

    # Linear combination search (against log_rmsd, more robust for skewed RMSD)
    candidates_conv = [m for m in candidate_metrics if m != "plddt"]

    def best_lin(metrics, y):
        X = np.column_stack([get(m) for m in metrics])
        X_c = X - X.mean(0); y_c = y - y.mean()
        coef, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
        pred = X_c @ coef
        r = pearson(pred, y_c)
        return r, coef

    print("\n=== best pairs (vs log_rmsd) ===")
    pair_results = []
    for a, b in itertools.combinations(candidates_conv, 2):
        r, coef = best_lin([a, b], log_rmsd)
        pair_results.append(((a, b), r, coef))
    pair_results.sort(key=lambda x: -abs(x[1]))
    for (a, b), r, coef in pair_results[:5]:
        print(f"  {a:>16} + {b:<16}  r={r:+.3f}")

    # Plot best individual vs pLDDT
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.28)
    fig.suptitle(f"Convexity metrics on 20-protein NATURAL benchmark (n={n})",
                 x=0.05, y=0.95, ha="left", fontsize=22, fontweight="bold")

    best_single = max(candidates_conv, key=lambda m: abs(pearson(get(m), log_rmsd)))
    best_single_r = pearson(get(best_single), log_rmsd)
    (pa, pb), pr, pcoef = pair_results[0]
    pair_x = pcoef[0] * (get(pa) - get(pa).mean()) + pcoef[1] * (get(pb) - get(pb).mean())

    plddt_r = pearson(get("plddt"), log_rmsd)

    panels = [
        ("pLDDT", get("plddt"), plddt_r),
        (best_single, get(best_single), best_single_r),
        (f"{pa} + {pb}", pair_x, pr),
    ]
    for ax, (lab, x, r) in zip(axes, panels):
        colors = ["#2f7d5a" if rr["rmsd"] < 5 else ("#ef6c00" if rr["rmsd"] < 15 else "#c62828")
                  for rr in rows]
        ax.scatter(x, log_rmsd, c=colors, s=130, edgecolors="#172033", lw=0.6)
        for row, xv in zip(rows, x):
            ax.annotate(row["name"], (xv, np.log10(max(row["rmsd"], 1e-3))),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axhline(np.log10(5),  color="#0f172a", lw=0.6, ls="--", alpha=0.4)
        ax.axhline(np.log10(15), color="#c62828", lw=0.6, ls="--", alpha=0.4)
        ax.text(0.04, 0.96, f"r = {r:+.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=13, fontweight="bold")
        ax.set_xlabel(lab); ax.set_ylabel("log₁₀(Cα RMSD, Å)"); ax.grid(True, lw=0.6)
        ax.set_title(lab, fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
