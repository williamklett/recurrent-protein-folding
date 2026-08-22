#!/usr/bin/env python3
"""Plot correct synthetic designs and hard synthetic misses on one separation chart."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#e2e8f0",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def load_rows(root: Path, group: str) -> list[dict]:
    rows = []
    for d in sorted(root.glob("*_analysis")):
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists():
            continue
        m = json.loads(metrics_p.read_text())
        corr_rows = list(csv.DictReader(corr_p.open()))
        layers = np.array([int(r["layer"]) for r in corr_rows])
        corr = np.array([float(r["corr_vs_47"]) for r in corr_rows])
        centroid = np.array([float(r["dist_mae_vs_centroid"]) for r in corr_rows])
        rows.append(
            {
                "name": d.name.removesuffix("_analysis"),
                "group": group,
                "analysis_dir": d,
                "length": int(m["length"]),
                "plddt": float(m["mean_plddt"]),
                "rmsd": float(m["ca_rmsd_after_kabsch"]),
                "mae": float(m["distance_mae_all_pairs"]),
                "tm_score": tm_score_for_analysis_dir(d),
                "dist_corr": float(m["distance_corr_all_pairs"]),
                "layers": layers,
                "corr": corr,
                "centroid": centroid,
                "min_last5": float(np.nanmin(corr[layers >= 43])),
                "late_amp": float(np.nanmax(corr[layers >= 36]) - np.nanmin(corr[layers >= 36])),
                "centroid_late_amp": float(np.nanmax(centroid[layers >= 36]) - np.nanmin(centroid[layers >= 36])),
            }
        )
    return rows


def tm_score_for_analysis_dir(d: Path) -> float:
    pred_p = d / "esmfold_direct_eval/esmfold_prediction_ca.npz"
    gt_p = d / "run_artifacts/gt_ca_distances.npz"
    if not pred_p.exists() or not gt_p.exists():
        return float("nan")
    pred = np.load(pred_p)["pred_ca"].astype(float)
    gt = np.load(gt_p)["ca_coords"].astype(float)
    n = min(len(pred), len(gt))
    if n < 3:
        return float("nan")
    pred = pred[:n]
    gt = gt[:n]
    pred_aligned = kabsch_align(pred, gt)
    distances = np.sqrt(((pred_aligned - gt) ** 2).sum(axis=1))
    d0 = 1.24 * np.cbrt(max(n - 15, 1)) - 1.8
    d0 = max(float(d0), 0.5)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(axis=0)
    tc = target - target.mean(axis=0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, det]) @ u.T
    return pc @ rot.T + target.mean(axis=0)


def corr_label(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 3 or np.nanstd(x) <= 1e-9 or np.nanstd(y) <= 1e-9:
        return ""
    return f"r = {float(np.corrcoef(x, y)[0, 1]):+.3f}"


def scatter_panel(
    ax: plt.Axes,
    rows: list[dict],
    x_key: str,
    x_label: str,
    title: str,
    y_key: str = "mae",
    y_label: str = "distance MAE vs PDB (A)",
    threshold_y: float = 2.0,
    threshold_label: str = "2 A miss threshold",
) -> None:
    groups = {
        "correct": {"label": "synthetic correct/tolerable", "color": "#2f7d5a", "marker": "o"},
        "miss": {"label": "hard synthetic miss", "color": "#d74b2f", "marker": "^"},
    }
    for group, spec in groups.items():
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        ax.scatter(
            [r[x_key] for r in g],
            [r[y_key] for r in g],
            s=95 if group == "correct" else 125,
            c=spec["color"],
            marker=spec["marker"],
            edgecolors="#172033",
            linewidths=0.75,
            alpha=0.92,
            label=f"{spec['label']} (n={len(g)})",
        )
        for r in g:
            annotate = r["group"] == "miss" or r["mae"] >= 1.55
            if annotate:
                ax.annotate(r["name"], (r[x_key], r[y_key]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    x = np.array([r[x_key] for r in rows])
    y = np.array([r[y_key] for r in rows])
    lab = corr_label(x, y)
    if lab:
        ax.text(0.04, 0.94, lab, transform=ax.transAxes, ha="left", va="top", fontsize=11)
    ax.axhline(threshold_y, color="#0f172a", lw=1.2, ls="--", alpha=0.45)
    ax.text(0.99, threshold_y, threshold_label, ha="right", va="bottom", fontsize=8, color="#334155")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, lw=0.7)


def transformed_rows(rows: list[dict]) -> list[dict]:
    out = []
    eps = 1e-6
    for row in rows:
        r = dict(row)
        r["log_one_minus_min_last5"] = float(np.log10(max(eps, 1.0 - r["min_last5"])))
        r["log_late_amp"] = float(np.log10(max(eps, r["late_amp"])))
        r["log_mae"] = float(np.log10(max(eps, r["mae"])))
        out.append(r)
    return out


def plot(rows: list[dict], out: Path, title: str) -> None:
    rows = transformed_rows(sorted(rows, key=lambda r: (r["group"] != "correct", r["mae"])))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9))
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.16, top=0.78, wspace=0.28)
    n_correct = sum(r["group"] == "correct" for r in rows)
    n_miss = sum(r["group"] == "miss" for r in rows)
    fig.suptitle(title, x=0.055, y=0.96, ha="left", fontsize=24, fontweight="bold", color="#172033")
    fig.text(
        0.055,
        0.89,
        f"Combined panel: {n_correct} correct/tolerable synthetic designs plus {n_miss} hard synthetic misses.",
        ha="left",
        fontsize=13,
        color="#475569",
    )
    scatter_panel(
        axes[0],
        rows,
        "plddt",
        "ESMFold mean pLDDT",
        "pLDDT calibration",
        y_key="tm_score",
        y_label="TM-score vs PDB",
        threshold_y=0.5,
        threshold_label="TM=0.5 fold threshold",
    )
    scatter_panel(
        axes[1],
        rows,
        "log_one_minus_min_last5",
        "log10(1 - min corr, last 5 blocks)",
        "convergence metric",
        y_key="tm_score",
        y_label="TM-score vs PDB",
        threshold_y=0.5,
        threshold_label="TM=0.5 fold threshold",
    )
    scatter_panel(
        axes[2],
        rows,
        "log_late_amp",
        "log10(late corr amplitude, blocks 36-47)",
        "late instability",
        y_key="tm_score",
        y_label="TM-score vs PDB",
        threshold_y=0.5,
        threshold_label="TM=0.5 fold threshold",
    )
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_summary(rows: list[dict], out: Path) -> None:
    fields = [
        "group",
        "name",
        "length",
        "mae",
        "tm_score",
        "rmsd",
        "dist_corr",
        "plddt",
        "min_last5",
        "late_amp",
        "centroid_late_amp",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["group"], x["mae"])):
            w.writerow({k: r[k] for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct-root", type=Path, required=True)
    ap.add_argument("--miss-root", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--title", default="Synthetic design separation benchmark")
    args = ap.parse_args()
    style()
    rows = load_rows(args.correct_root, "correct") + load_rows(args.miss_root, "miss")
    if not rows:
        raise SystemExit("No rows found")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.out_prefix.with_name(args.out_prefix.name + "_separation_scatter.png"), args.title)
    write_summary(rows, args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"))
    print(f"N={len(rows)} correct={sum(r['group']=='correct' for r in rows)} miss={sum(r['group']=='miss' for r in rows)}")
    for r in sorted(rows, key=lambda x: x["mae"]):
        print(
            f"{r['group']:>7} {r['name']:>6} MAE={r['mae']:.2f} "
            f"pLDDT={r['plddt']:.3f} min_last5={r['min_last5']:.3f} late_amp={r['late_amp']:.4f}"
        )


if __name__ == "__main__":
    main()
