#!/usr/bin/env python3
"""Plot convergence diagnostics with TM-score on the y-axis."""

from __future__ import annotations

import argparse
import csv
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


def tm_score(run_or_analysis_dir: Path) -> float:
    if (run_or_analysis_dir / "esmfold_direct_eval").exists():
        pred_p = run_or_analysis_dir / "esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_candidates = [
            run_or_analysis_dir / "run_artifacts/gt_ca_distances.npz",
            run_or_analysis_dir.parent / "gt_ca_distances.npz",
        ]
    else:
        pred_p = run_or_analysis_dir / "analysis/esmfold_direct_eval/esmfold_prediction_ca.npz"
        gt_candidates = [
            run_or_analysis_dir / "gt_ca_distances.npz",
            run_or_analysis_dir / "run_artifacts/gt_ca_distances.npz",
        ]
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


def load_rows(summary_csv: Path, tm_root: Path) -> list[dict]:
    rows = []
    for row in csv.DictReader(summary_csv.open()):
        name = row["name"]
        candidates = [
            tm_root / f"{name}_analysis",
            tm_root / f"benchmark_{name}",
            tm_root / f"hard_synthetic_miss_{name}",
            tm_root / "general_convergence_n40_native_cw30_partial" / f"{name}_analysis",
            tm_root / "general_convergence_n40_native" / f"{name}_analysis",
            tm_root / "general_convergence_hard_synthetic_native" / f"{name}_analysis",
            tm_root / "partial_benchmark" / f"{name}_analysis",
            tm_root / "hard_synthetic_miss_partial_sync" / f"{name}_analysis",
            tm_root / "synthetic_design_partial_sync" / f"{name}_analysis",
        ]
        analysis_dir = next((p for p in candidates if p.exists()), candidates[0])
        min_last5 = float(row["min_last5"])
        late_amp = float(row["late_amp"])
        rows.append(
            {
                "name": name,
                "mae": float(row["mae"]),
                "plddt": float(row["plddt"]),
                "tm_score": tm_score(analysis_dir),
                "log_residual": float(np.log10(max(1.0 - min_last5, 1e-6))),
                "log_late_amp": float(np.log10(max(late_amp, 1e-6))),
            }
        )
    rows = [r for r in rows if np.isfinite(r["tm_score"])]
    rows.sort(key=lambda r: r["tm_score"], reverse=True)
    return rows


def corr_text(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 3 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return ""
    return f"r = {float(np.corrcoef(x, y)[0, 1]):+.3f}"


def plot(rows: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.14, top=0.80, wspace=0.28)
    fig.suptitle(title, x=0.055, y=0.96, ha="left", fontsize=24, fontweight="bold", color="#172033")
    fig.text(0.055, 0.89, f"N={len(rows)} cached proteins. Y-axis is TM-score vs PDB.", ha="left", fontsize=13, color="#475569")

    y = np.array([r["tm_score"] for r in rows])
    names = [r["name"] for r in rows]
    panels = [
        (np.array([r["plddt"] for r in rows]), "ESMFold mean pLDDT", "pLDDT calibration"),
        (np.array([r["log_residual"] for r in rows]), "log10(1 - min corr), final 5 readouts", "convergence classifier"),
        (np.array([r["log_late_amp"] for r in rows]), "log10(late corr amplitude), readouts 35-46", "instability classifier"),
    ]
    colors = mpl.colormaps["RdYlGn"](np.linspace(0.15, 0.85, len(rows)))[::-1]
    for ax, (x, xlabel, subtitle) in zip(axes, panels):
        ax.scatter(x, y, s=115, c=colors, edgecolors="#172033", linewidths=0.7)
        for name, xx, yy in zip(names, x, y):
            ax.annotate(name, (xx, yy), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0.5, color="#64748b", ls="--", lw=1.2)
        ax.text(0.98, 0.5, "TM=0.5", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color="#475569")
        ax.text(0.04, 0.94, corr_text(x, y), transform=ax.transAxes, ha="left", va="top", fontsize=11)
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TM-score vs PDB")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, lw=0.7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", type=Path, required=True)
    ap.add_argument("--tm-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Convergence diagnostics vs TM-score")
    args = ap.parse_args()
    style()
    rows = load_rows(args.summary_csv, args.tm_root)
    if not rows:
        raise SystemExit("No rows with computable TM-score")
    plot(rows, args.out, args.title)
    print(f"N={len(rows)}")
    for r in rows:
        print(f"{r['name']:>6} TM={r['tm_score']:.3f} MAE={r['mae']:.2f} pLDDT={r['plddt']:.3f}")


if __name__ == "__main__":
    main()
