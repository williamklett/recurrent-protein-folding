#!/usr/bin/env python3
"""Analyze residual-decay shape from layerwise convergence curves."""

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


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(axis=0)
    tc = target - target.mean(axis=0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, det]) @ u.T
    return pc @ rot.T + target.mean(axis=0)


def tm_score(name: str, tm_root: Path) -> float:
    candidates = [
        tm_root / f"benchmark_{name}",
        tm_root / f"hard_synthetic_miss_{name}",
        tm_root / "general_convergence_n40_native_cw30_partial" / f"{name}_analysis",
        tm_root / "general_convergence_n40_native" / f"{name}_analysis",
        tm_root / "general_convergence_hard_synthetic_native" / f"{name}_analysis",
        tm_root / "hard_synthetic_miss_partial_sync" / f"{name}_analysis",
        tm_root / "partial_benchmark" / f"{name}_analysis",
        tm_root / "synthetic_design_partial_sync" / f"{name}_analysis",
    ]
    root = next((p for p in candidates if p.exists()), candidates[0])
    if (root / "esmfold_direct_eval").exists():
        pred_p = root / "esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_candidates = [root / "run_artifacts/gt_ca_distances.npz", root.parent / "gt_ca_distances.npz"]
    else:
        pred_p = root / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_candidates = [root / "gt_ca_distances.npz", root / "run_artifacts/gt_ca_distances.npz"]
    gt_p = next((p for p in gt_candidates if p.exists()), gt_candidates[0])
    if not pred_p.exists() or not gt_p.exists():
        return float("nan")
    pred = np.load(pred_p)["pred_ca"].astype(float)
    gt = np.load(gt_p)["ca_coords"].astype(float)
    n = min(len(pred), len(gt))
    if n < 3:
        return float("nan")
    pred = pred[:n]
    gt = gt[:n]
    aligned = kabsch_align(pred, gt)
    distances = np.sqrt(((aligned - gt) ** 2).sum(axis=1))
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def metrics_for_name(name: str, root: Path) -> dict | None:
    candidates = [
        root / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / f"benchmark_{name}" / "analysis/esmfold_direct_eval/metrics.json",
        Path("runs") / f"hard_synthetic_miss_{name}" / "analysis/esmfold_direct_eval/metrics.json",
        Path("runs") / "general_convergence_n40_native_cw30_partial" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "general_convergence_n40_native" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "general_convergence_hard_synthetic_native" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "partial_benchmark" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "hard_synthetic_miss_partial_sync" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
        Path("runs") / "synthetic_design_partial_sync" / f"{name}_analysis" / "esmfold_direct_eval/metrics.json",
    ]
    p = next((p for p in candidates if p.exists()), None)
    return json.loads(p.read_text()) if p else None


def row_from_curve(name: str, corr_p: Path, metrics: dict, tm_root: Path) -> dict:
    corr_rows = list(csv.DictReader(corr_p.open()))
    layers = np.array([int(r["layer"]) for r in corr_rows])
    key = "corr_vs_final_distogram" if "corr_vs_final_distogram" in corr_rows[0] else "corr_vs_47"
    corr = np.array([float(r[key]) for r in corr_rows])
    # Drop the reference row if present; shape should be measured on probe outputs.
    keep = layers < int(np.nanmax(layers)) if int(np.nanmax(layers)) == 47 else layers <= 46
    cm = curve_metrics(corr[keep], layers[keep])
    return {
        "name": name,
        "group": "natural" if not name[0].isdigit() else "synthetic",
        "length": int(metrics["length"]),
        "mae": float(metrics["distance_mae_all_pairs"]),
        "log_mae": float(np.log10(max(metrics["distance_mae_all_pairs"], 1e-8))),
        "plddt": float(metrics["mean_plddt"]),
        "tm_score": tm_score(name, tm_root),
        **cm,
    }


def fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def curve_metrics(corr: np.ndarray, layers: np.ndarray) -> dict[str, float]:
    residual = np.clip(1.0 - corr, 1e-8, 1.0)
    log_residual = np.log10(residual)
    slope, r2 = fit_r2(layers.astype(float), log_residual)
    delta = np.diff(residual)
    backtrack_mass = float(np.maximum(delta, 0.0).sum())
    backtrack_steps = int((delta > 0).sum())
    auc = float(np.trapezoid(residual, layers))
    late = layers >= max(layers) - 11
    late_slope, late_r2 = fit_r2(layers[late].astype(float), log_residual[late])
    return {
        "log_resid_slope": slope,
        "log_resid_r2": r2,
        "late_log_resid_slope": late_slope,
        "late_log_resid_r2": late_r2,
        "residual_auc": auc,
        "backtrack_mass": backtrack_mass,
        "backtrack_steps": backtrack_steps,
        "start_residual": float(residual[0]),
        "final_probe_residual": float(residual[-2] if len(residual) > 1 and corr[-1] >= 0.999999 else residual[-1]),
    }


def load_rows(root: Path, tm_root: Path) -> list[dict]:
    rows = []
    flat_csvs = sorted(root.glob("*_layerwise.csv"))
    if flat_csvs:
        for corr_p in flat_csvs:
            name = corr_p.name.removesuffix("_layerwise.csv")
            metrics = metrics_for_name(name, root)
            if metrics is None:
                continue
            rows.append(row_from_curve(name, corr_p, metrics, tm_root))
        rows.sort(key=lambda r: r["mae"])
        return rows

    for d in sorted(root.glob("*_analysis")):
        name = d.name.removesuffix("_analysis")
        metrics_p = d / "esmfold_direct_eval/metrics.json"
        corr_p = d / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists():
            continue
        metrics = json.loads(metrics_p.read_text())
        rows.append(row_from_curve(name, corr_p, metrics, tm_root))
    rows.sort(key=lambda r: r["mae"])
    return rows


def corr_label(x: np.ndarray, y: np.ndarray) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) <= 1e-12 or np.nanstd(y[mask]) <= 1e-12:
        return ""
    return f"r = {float(np.corrcoef(x[mask], y[mask])[0, 1]):+.3f}"


def plot(rows: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.08, top=0.86, wspace=0.28, hspace=0.35)
    fig.suptitle(title, x=0.055, y=0.965, ha="left", fontsize=24, fontweight="bold", color="#172033")
    fig.text(0.055, 0.915, f"N={len(rows)}. Residual is 1 - corr(readout, native ESMFold output distogram).", ha="left", fontsize=13, color="#475569")

    colors = ["#287d52" if r["mae"] < 2.0 else "#e99a24" if r["mae"] < 5.0 else "#c9452a" for r in rows]
    names = [r["name"] for r in rows]
    panels = [
        ("log_resid_r2", "log-residual linear fit R²", "log10(MAE)", "log_mae"),
        ("log_resid_slope", "log-residual slope", "log10(MAE)", "log_mae"),
        ("backtrack_mass", "backtracking mass", "log10(MAE)", "log_mae"),
        ("residual_auc", "residual AUC", "TM-score", "tm_score"),
        ("late_log_resid_r2", "late log-residual fit R²", "TM-score", "tm_score"),
        ("backtrack_steps", "backtracking steps", "TM-score", "tm_score"),
    ]
    for ax, (xkey, xlabel, ylabel, ykey) in zip(axes.flat, panels):
        x = np.array([r[xkey] for r in rows], dtype=float)
        y = np.array([r[ykey] for r in rows], dtype=float)
        ax.scatter(x, y, s=95, c=colors, edgecolors="#172033", linewidths=0.7, alpha=0.95)
        for name, xx, yy in zip(names, x, y):
            if np.isfinite(xx) and np.isfinite(yy):
                ax.annotate(name, (xx, yy), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        ax.text(0.04, 0.94, corr_label(x, y), transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_csv(rows: list[dict], out: Path) -> None:
    fields = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tm-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--title", default="Residual decay shape diagnostics")
    args = parser.parse_args()
    style()
    rows = load_rows(args.root, args.tm_root)
    if not rows:
        raise SystemExit("No analyzable rows")
    write_csv(rows, args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"))
    plot(rows, args.out_prefix.with_name(args.out_prefix.name + "_panel.png"), args.title)
    print(f"N={len(rows)}")
    for key in ["log_resid_r2", "log_resid_slope", "backtrack_mass", "residual_auc", "late_log_resid_r2"]:
        x = np.array([r[key] for r in rows], dtype=float)
        y = np.array([r["log_mae"] for r in rows], dtype=float)
        print(key, corr_label(x, y))


if __name__ == "__main__":
    main()
