#!/usr/bin/env python3
"""Test whether convexity metrics add orthogonal information beyond pLDDT.

For each benchmark and target:
  • Fit pLDDT alone → r_plddt (baseline)
  • Fit pLDDT + candidate metric → r_combined
  • Report Δr  and partial-correlation gain
  • Also report the "residual correlation" — correlation of metric with the
    residual after regressing target on pLDDT alone. This is the cleanest
    test of orthogonality.

If a metric's r_combined > r_plddt AND residual_r is large, the metric is
catching cases pLDDT misses — a clean "additional information" result.
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


def pearson(x, y):
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_lin(X: np.ndarray, y: np.ndarray):
    """OLS, no intercept (centered). Return predictions."""
    X_c = X - X.mean(axis=0); y_c = y - y.mean()
    coef, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
    return X_c @ coef + y.mean(), coef


def analyze(rows, target_key, target_label, candidate_metrics, name="", out=None):
    print(f"\n=== {name} (n={len(rows)}, target={target_label}) ===")
    y = np.array([r[target_key] for r in rows])
    if target_key == "rmsd":
        y_t = np.log10(np.maximum(y, 1e-3))
        y_label = "log10(RMSD)"
    else:
        y_t = y
        y_label = target_label
    plddt = np.array([r["plddt"] for r in rows])

    r_plddt = abs(pearson(plddt, y_t))
    print(f"  baseline pLDDT alone:  |r| = {r_plddt:.3f}")

    # Residual after pLDDT
    resid_t = y_t - fit_lin(plddt[:, None], y_t)[0]

    results = []
    for m in candidate_metrics:
        x = np.array([r[m] for r in rows])
        r_alone = abs(pearson(x, y_t))
        r_resid = pearson(x, resid_t)   # signed
        r_combined = abs(pearson(fit_lin(np.column_stack([plddt, x]), y_t)[0], y_t))
        results.append({
            "metric": m, "r_alone": r_alone, "r_resid": r_resid,
            "r_combined": r_combined, "delta": r_combined - r_plddt,
        })
    results.sort(key=lambda r: -r["delta"])

    print(f"\n  {'metric':>20}  {'|r| alone':>10}  {'|r| combined':>13}  {'Δ over pLDDT':>13}  {'r_resid':>10}")
    for res in results:
        print(f"  {res['metric']:>20}  {res['r_alone']:>+10.3f}  "
              f"{res['r_combined']:>+13.3f}  {res['delta']:>+13.3f}  {res['r_resid']:>+10.3f}")

    if out is None: return r_plddt, results

    # Plot top 3 by Δ
    top3 = results[:3]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.07, wspace=0.27, hspace=0.35)
    fig.suptitle(f"pLDDT + convexity on {name} (n={len(rows)})",
                 x=0.05, y=0.96, ha="left", fontsize=22, fontweight="bold")

    # Row 1: pLDDT-vs-target, then top-2 metrics combined-with-pLDDT
    panels = [{"label": f"pLDDT alone  |r|={r_plddt:.3f}",
               "x_lab": "pLDDT", "x": plddt}]
    for res in top3:
        m = res["metric"]
        x = np.array([r[m] for r in rows])
        combined, coef = fit_lin(np.column_stack([plddt, x]), y_t)
        panels.append({
            "label": f"pLDDT + {m}  |r|={res['r_combined']:.3f}",
            "x_lab": f"pLDDT + {m}",
            "x": combined,
        })

    for ax, p in zip(axes[0], panels[:3]):
        ax.scatter(p["x"], y_t, c="#2563eb", s=100, edgecolors="#172033", lw=0.5)
        for r, xv in zip(rows, p["x"]):
            ax.annotate(r.get("name", ""), (xv, y_t[rows.index(r) if r in rows else 0]),
                        fontsize=7, xytext=(3,3), textcoords="offset points")
        ax.set_xlabel(p["x_lab"]); ax.set_ylabel(y_label); ax.grid(True, lw=0.5)
        ax.set_title(p["label"], fontsize=11)

    # Row 2: residual scatter — metric vs (target after pLDDT removed)
    for ax, res in zip(axes[1], top3):
        m = res["metric"]
        x = np.array([r[m] for r in rows])
        ax.scatter(x, resid_t, c="#7c3aed", s=100, edgecolors="#172033", lw=0.5)
        for r, xv in zip(rows, x):
            yi = list(rows).index(r)
            ax.annotate(r.get("name", ""), (xv, resid_t[yi]),
                        fontsize=7, xytext=(3,3), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_xlabel(m); ax.set_ylabel(f"{y_label} − pLDDT_predicted")
        ax.set_title(f"residual: {m}  r={res['r_resid']:+.3f}", fontsize=11)
        ax.grid(True, lw=0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\n  wrote {out}")
    return r_plddt, results


def load_synthetic_csv(p: Path):
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        for k, v in r.items():
            if k not in ("name", "group"):
                try: r[k] = float(v)
                except: r[k] = float("nan")
    rows = [r for r in rows if not np.isnan(r.get("tm", float("nan")))]
    return rows


def convexity_for_natural(root: Path):
    """Compute convexity metrics for natural benchmark with same definitions."""
    from convexity_natural_benchmark import load_rows
    return load_rows(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-csv", type=Path, required=True)
    ap.add_argument("--natural-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    style()

    candidates = [
        "ttc98", "ttc99",
        "log_late_amp", "log_neg_step", "log_auc_late",
        "log_max_drawdown", "log_deficit_area", "log_d2_std",
    ]

    # Synthetic (TM as target)
    syn_rows = load_synthetic_csv(args.synthetic_csv)
    syn_r_base, syn_results = analyze(
        syn_rows, target_key="tm", target_label="TM-score",
        candidate_metrics=candidates,
        name="Synthetic OOD",
        out=args.out_dir / "plddt_plus_convexity_synthetic.png",
    )

    # Natural (RMSD as target)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convexity_natural_benchmark import load_rows as load_nat
    nat_rows = load_nat(args.natural_root)
    nat_r_base, nat_results = analyze(
        nat_rows, target_key="rmsd", target_label="Cα RMSD",
        candidate_metrics=candidates,
        name="Natural",
        out=args.out_dir / "plddt_plus_convexity_natural.png",
    )

    # Cross-domain summary: which metric helps in BOTH?
    print("\n\n=== CROSS-DOMAIN  Δ |r| over pLDDT-alone ===")
    print(f"  {'metric':>20}  {'synthetic Δ':>13}  {'natural Δ':>11}  {'avg':>7}")
    syn_d = {r["metric"]: r["delta"] for r in syn_results}
    nat_d = {r["metric"]: r["delta"] for r in nat_results}
    rows_x = [(m, syn_d[m], nat_d[m], (syn_d[m] + nat_d[m]) / 2) for m in candidates]
    rows_x.sort(key=lambda x: -x[3])
    for m, s, n, a in rows_x:
        print(f"  {m:>20}  {s:>+13.3f}  {n:>+11.3f}  {a:>+7.3f}")


if __name__ == "__main__":
    main()
