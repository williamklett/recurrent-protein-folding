#!/usr/bin/env python3
"""
Aggregate the overnight benchmark: pull metrics from every runs/benchmark_*/
directory, produce the headline figure (convergence metric vs pLDDT vs accuracy),
and compute the cross-protein statistics.

The per-protein outputs from benchmark_one_protein.sh land in:
  runs/benchmark_<name>/
    analysis/
      esmfold_direct_eval/metrics.json   (pLDDT, RMSD, distance_corr)
      layerwise_distance_correlations.csv
      mds_structure_video_fixed/distogram_mds_models.pdb

This script reads all of those and produces:
  • runs/benchmark_summary.csv        — per-protein table
  • runs/benchmark_summary.json       — same, machine-readable
  • runs/benchmark_fig1_ttc_vs_plddt.png    — headline scatter
  • runs/benchmark_fig2_three_panel.png     — TTC vs RMSD, vs distMAE, vs distcorr
  • runs/benchmark_fig3_calibration.png     — confidence calibration: where does pLDDT mis-fire vs TTC
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent


def smooth(x, k=5):
    pad = k // 2
    return np.array([x[max(0,i-pad):min(len(x),i+pad+1)].mean() for i in range(len(x))])


def compute_ttc(corr_curve, threshold=0.99):
    above = np.where(corr_curve > threshold)[0]
    return int(above[0]) + 1 if len(above) > 0 else None


def main():
    bench_dirs = sorted(ROOT.glob("runs/benchmark_*/"))
    if not bench_dirs:
        print(f"No runs/benchmark_*/ directories found under {ROOT}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for d in bench_dirs:
        name = d.name.replace("benchmark_", "")
        metrics_p = d / "analysis" / "esmfold_direct_eval" / "metrics.json"
        corr_p    = d / "analysis" / "layerwise_distance_correlations.csv"
        if not metrics_p.exists() or not corr_p.exists():
            print(f"  [skip] {name}: missing analysis files")
            continue
        m = json.loads(metrics_p.read_text())

        with open(corr_p) as f:
            reader = csv.DictReader(f)
            curve_raw = np.array([float(r["corr_vs_47"]) for r in reader])
        if len(curve_raw) < 30:
            print(f"  [skip] {name}: only {len(curve_raw)} layers")
            continue
        curve_smooth = smooth(curve_raw, k=5)

        row = {
            "name":              name,
            "length":            int(m["length"]),
            "mean_plddt":        float(m["mean_plddt"]),
            "ca_rmsd":           float(m["ca_rmsd_after_kabsch"]),
            "distance_mae":      float(m["distance_mae_all_pairs"]),
            "distance_corr":     float(m["distance_corr_all_pairs"]),
            "corr_b30_b47":      float(curve_smooth[29]),
            "log_one_minus":     float(np.log10(max(1 - curve_smooth[29], 1e-9))),
            "ttc_0p99":          compute_ttc(curve_smooth, 0.99) or 99,
            "ttc_0p98":          compute_ttc(curve_smooth, 0.98) or 99,
        }
        rows.append(row)

    if not rows:
        print("No usable runs.", file=sys.stderr); sys.exit(1)

    rows.sort(key=lambda r: r["ttc_0p99"])

    # ── Per-protein table ────────────────────────────────────────────────────
    print(f"\n{'protein':>14}  {'L':>4}  {'pLDDT':>6}  {'RMSD':>7}  {'distMAE':>8}  "
          f"{'distCorr':>9}  {'corr(30,47)':>11}  {'TTC>0.99':>9}")
    print("─" * 90)
    for r in rows:
        print(f"{r['name']:>14}  {r['length']:>4}  {r['mean_plddt']:6.3f}  "
              f"{r['ca_rmsd']:7.2f}  {r['distance_mae']:8.2f}  "
              f"{r['distance_corr']:9.3f}  {r['corr_b30_b47']:11.4f}  {r['ttc_0p99']:>9}")

    # ── Save CSV + JSON ──────────────────────────────────────────────────────
    csv_path  = ROOT / "runs" / "benchmark_summary.csv"
    json_path = ROOT / "runs" / "benchmark_summary.json"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {csv_path} and {json_path}")

    # ── Correlations ─────────────────────────────────────────────────────────
    arr = lambda k: np.array([r[k] for r in rows], dtype=float)
    print(f"\nCross-protein correlations (N={len(rows)}):")
    for x_name in ["mean_plddt", "ttc_0p99", "log_one_minus"]:
        x = arr(x_name)
        for y_name in ["ca_rmsd", "distance_mae", "distance_corr"]:
            y = arr(y_name)
            if x.std() < 1e-9 or y.std() < 1e-9: continue
            r = float(np.corrcoef(x, y)[0,1])
            print(f"  r({x_name:>14}, {y_name:<13}) = {r:+.3f}")

    # ── Headline figure: TTC vs pLDDT, colored by accuracy ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rmsd_arr = arr("ca_rmsd")
    norm = plt.Normalize(vmin=0, vmax=max(40, rmsd_arr.min()*2))
    cmap = plt.cm.viridis_r   # darker = more damage

    # Panel A: pLDDT vs accuracy
    ax = axes[0]
    sc = ax.scatter(arr("mean_plddt"), rmsd_arr, c=rmsd_arr, cmap=cmap, norm=norm,
                    s=140, edgecolors="black", lw=0.7)
    for r in rows:
        ax.annotate(r["name"], (r["mean_plddt"], r["ca_rmsd"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0.7, ls="--", color="gray", lw=0.7, label="pLDDT trust threshold")
    ax.set_xlabel("ESMFold mean pLDDT"); ax.set_ylabel("Cα RMSD vs ground truth (Å)")
    ax.set_title(f"pLDDT vs accuracy  (N={len(rows)})")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # Panel B: TTC vs accuracy
    ax = axes[1]
    ax.scatter(arr("ttc_0p99"), rmsd_arr, c=rmsd_arr, cmap=cmap, norm=norm,
               s=140, edgecolors="black", lw=0.7)
    for r in rows:
        ax.annotate(r["name"], (r["ttc_0p99"], r["ca_rmsd"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(30, ls="--", color="gray", lw=0.7, label="TTC threshold")
    ax.set_xlabel("Time-to-convergence (first block, smoothed corr > 0.99)")
    ax.set_ylabel("Cα RMSD vs ground truth (Å)")
    ax.set_title("TTC (our metric) vs accuracy")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # Panel C: pLDDT vs TTC (are they redundant?)
    ax = axes[2]
    sc = ax.scatter(arr("mean_plddt"), arr("ttc_0p99"), c=rmsd_arr, cmap=cmap, norm=norm,
                    s=140, edgecolors="black", lw=0.7)
    for r in rows:
        ax.annotate(r["name"], (r["mean_plddt"], r["ttc_0p99"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("pLDDT"); ax.set_ylabel("TTC")
    ax.set_title("pLDDT vs TTC — color = RMSD")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cα RMSD (Å)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = ROOT / "runs" / "benchmark_fig1_ttc_vs_plddt.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"\nSaved {p}")

    # ── Calibration plot: where does pLDDT miss-fire vs TTC ──────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    # Quadrants: pLDDT < 0.7 vs ≥ 0.7  ×  TTC < 30 vs ≥ 30
    for r in rows:
        is_correct = r["ca_rmsd"] < 15.0  # heuristic threshold for "correct"
        marker = "o" if is_correct else "X"
        color = "#1976D2" if is_correct else "#D32F2F"
        ax.scatter(r["mean_plddt"], r["ttc_0p99"], s=160, marker=marker,
                   color=color, edgecolors="black", lw=0.7)
        ax.annotate(r["name"], (r["mean_plddt"], r["ttc_0p99"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.axhline(30, ls="--", color="gray", lw=0.7)
    ax.axvline(0.7, ls="--", color="gray", lw=0.7)
    ax.set_xlabel("pLDDT"); ax.set_ylabel("TTC")
    ax.set_title("Confidence calibration: ● correct (RMSD<15Å), ✗ failed (RMSD≥15Å)")
    ax.text(0.99, 0.99, "  upper-right = both flag failure\nupper-left  = TTC catches what pLDDT misses\nlower-right = pLDDT catches what TTC misses\nlower-left  = both say converged",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = ROOT / "runs" / "benchmark_fig3_calibration.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
