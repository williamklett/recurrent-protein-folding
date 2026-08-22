#!/usr/bin/env python3
"""Combined probe metric: z-score(pLDDT) - z-score(pce_raw) - z-score(bt_raw).

Each constituent is z-score normalised over the full dataset.
pLDDT contributes positively (higher = better).
pce_raw and bt_raw contribute negatively (higher = worse fold).

Plots:
  Row 0: all N proteins
  Row 1: TM < 0.5 failures only

Columns:
  0 — pLDDT alone (baseline)
  1 — combined score
  2 — combined vs pLDDT scatter (to show orthogonality)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT    = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "runs/general_convergence_ood500_native_cw30_stream"


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def kadane_max(arr: np.ndarray) -> float:
    best = cur = 0.0
    for v in arr:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def pce_raw(corr: np.ndarray) -> float:
    d2 = np.diff(corr.astype(float), n=2)
    return kadane_max(np.maximum(d2, 0.0))


def backtrack_mass(corr: np.ndarray) -> float:
    residual = np.clip(1.0 - corr, 1e-8, 1.0)
    delta = np.diff(residual)
    return float(np.maximum(delta, 0.0).sum())


def load_data(ood_dir: Path) -> list[dict]:
    summary = {r["protein"]: r for r in csv.DictReader(
        (ood_dir / "general_convergence_summary.csv").open())}

    data = []
    for d in sorted(ood_dir.glob("*_analysis")):
        corr_p = d / "layerwise_distance_correlations.csv"
        if not corr_p.exists():
            continue
        name = d.name.removesuffix("_analysis")
        s = summary.get(name, {})
        if not s:
            continue
        tm    = float(s.get("tm_score", "nan"))
        plddt = float(s.get("plddt",    "nan"))
        if np.isnan(tm) or np.isnan(plddt):
            continue

        rows   = list(csv.DictReader(corr_p.open()))
        key    = "corr_vs_final_distogram" if "corr_vs_final_distogram" in rows[0] else "corr_vs_47"
        layers = np.array([int(r["layer"]) for r in rows])
        corr   = np.array([float(r[key])   for r in rows])
        corr46 = corr[layers <= 46]

        data.append({
            "name":    name,
            "tm":      tm,
            "plddt":   plddt,
            "pce_raw": pce_raw(corr46),
            "bt_raw":  backtrack_mass(corr46),
        })
    return data


def zscore(arr: np.ndarray) -> np.ndarray:
    mu, sd = arr.mean(), arr.std()
    return (arr - mu) / (sd + 1e-12)


def tm_color(tm: float) -> str:
    if tm >= 0.7: return "#287d52"
    if tm >= 0.5: return "#e99a24"
    return "#c9452a"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3: return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def draw_scatter(ax, x, y, names, xlabel, ylabel, title, fail_thresh=0.5,
                 annotate=True):
    colors = [tm_color(tm) for tm in y]
    ax.scatter(x, y, s=95, c=colors, edgecolors="#172033",
               linewidths=0.6, alpha=0.92, zorder=3)
    if annotate:
        for nm, xx, yy in zip(names, x, y):
            if np.isfinite(xx) and np.isfinite(yy):
                ax.annotate(nm, (xx, yy), fontsize=7, xytext=(4, 3),
                            textcoords="offset points", color="#1e293b")
    if ylabel == "TM-score vs PDB":
        ax.axhline(fail_thresh, color="#0f172a", lw=0.8, ls="--", alpha=0.4)
    r = pearson_r(np.asarray(x, float), np.asarray(y, float))
    ax.text(0.04, 0.96, f"r = {r:+.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fef3c7", ec="#a16207", alpha=0.92))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    if ylabel == "TM-score vs PDB":
        ax.set_ylim(-0.05, 1.05)
    ax.grid(True, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def auroc(scores: np.ndarray, fail_labels: np.ndarray) -> float:
    n_pos = fail_labels.sum(); n_neg = len(fail_labels) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    thresholds = np.linspace(scores.min(), scores.max(), 600)
    fprs, tprs = [], []
    for t in thresholds:
        pred = scores >= t
        tprs.append((pred & fail_labels).sum() / n_pos)
        fprs.append((pred & ~fail_labels).sum() / n_neg)
    idx = np.argsort(fprs)
    return float(np.trapezoid(np.array(tprs)[idx], np.array(fprs)[idx]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir",     type=Path, default=OOD_DIR)
    ap.add_argument("--out",         type=Path, default=OOD_DIR / "combined_metric_vs_tm.png")
    ap.add_argument("--fail-thresh", type=float, default=0.5)
    args = ap.parse_args()

    style()
    data = load_data(args.ood_dir)
    print(f"N={len(data)}")

    # Build combined score on the full dataset for z-score calibration
    plddt_all  = np.array([r["plddt"]   for r in data])
    pce_all    = np.array([r["pce_raw"] for r in data])
    bt_all     = np.array([r["bt_raw"]  for r in data])
    tm_all     = np.array([r["tm"]      for r in data])

    z_plddt = zscore(plddt_all)
    z_pce   = zscore(pce_all)
    z_bt    = zscore(bt_all)
    combined = z_plddt - z_pce - z_bt          # higher = better

    for i, r in enumerate(data):
        r["combined"] = float(combined[i])

    fail = [r for r in data if r["tm"] < args.fail_thresh]
    print(f"failures(TM<{args.fail_thresh})={len(fail)}")

    # AUROC
    fail_labels = (tm_all < args.fail_thresh)
    auc_plddt    = auroc(-plddt_all,          fail_labels)
    auc_pce      = auroc(pce_all,             fail_labels)
    auc_bt       = auroc(bt_all,              fail_labels)
    auc_combined = auroc(-combined,           fail_labels)

    print(f"\nAUROC (TM<{args.fail_thresh} = failure):")
    print(f"  pLDDT alone : {auc_plddt:.4f}")
    print(f"  pce_raw     : {auc_pce:.4f}")
    print(f"  bt_raw      : {auc_bt:.4f}")
    print(f"  combined    : {auc_combined:.4f}")

    print(f"\nPearson r with TM-score:")
    for subset, label in [(data, "all"), (fail, "TM<0.5")]:
        tm_s = np.array([r["tm"] for r in subset])
        for key in ("plddt", "pce_raw", "bt_raw", "combined"):
            x = np.array([r[key] for r in subset])
            print(f"  [{label:7s}] {key:10s}: r = {pearson_r(x, tm_s):+.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.87, bottom=0.08,
                        wspace=0.30, hspace=0.40)
    fig.suptitle("Combined probe metric: z(pLDDT) − z(pce_raw) − z(bt_raw)",
                 x=0.05, y=0.96, ha="left", fontsize=20,
                 fontweight="bold", color="#172033")
    fig.text(
        0.05, 0.915,
        f"N={len(data)} OOD-500 proteins.  "
        f"Combined = z-score(pLDDT) − z-score(pce_raw) − z-score(bt_raw); higher = more confident fold.  "
        f"AUROC — pLDDT: {auc_plddt:.3f}  pce_raw: {auc_pce:.3f}  "
        f"bt_raw: {auc_bt:.3f}  combined: {auc_combined:.3f}",
        ha="left", fontsize=11, color="#475569",
    )

    for row_idx, (subset, row_label) in enumerate([
        (data, f"All proteins  (N={len(data)})"),
        (fail, f"Fold failures only — TM < {args.fail_thresh}  (N={len(fail)})"),
    ]):
        tm_s   = np.array([r["tm"]       for r in subset])
        plddt_s = np.array([r["plddt"]   for r in subset])
        comb_s  = np.array([r["combined"] for r in subset])
        names   = [r["name"]              for r in subset]

        draw_scatter(axes[row_idx, 0], plddt_s, tm_s, names,
                     "ESMFold mean pLDDT", "TM-score vs PDB",
                     f"pLDDT baseline — {row_label}", args.fail_thresh)

        draw_scatter(axes[row_idx, 1], comb_s, tm_s, names,
                     "z(pLDDT) − z(pce_raw) − z(bt_raw)", "TM-score vs PDB",
                     f"Combined score — {row_label}", args.fail_thresh)

        draw_scatter(axes[row_idx, 2], plddt_s, comb_s, names,
                     "ESMFold mean pLDDT", "Combined score",
                     f"pLDDT vs combined — {row_label}",
                     annotate=(row_idx == 1))  # annotate only failure subset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
