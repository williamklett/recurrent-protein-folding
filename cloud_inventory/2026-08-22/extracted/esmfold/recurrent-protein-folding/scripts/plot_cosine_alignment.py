#!/usr/bin/env python3
"""Plot cosine alignment between Δz_t and −∇L for the three candidate losses,
color-coded by success (RMSD < 5 Å) vs failure.

For the GD claim:
  - cos_neg_grad_gt should be POSITIVE for successes, ~zero for failures
    (failures don't have the ground truth so can't be descending it)
  - cos_neg_grad_fin should be POSITIVE for everyone (model converges to its own
    final answer in both cases)
  - cos_random should be ~zero everywhere
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
COS_DIR = ROOT / "runs/cosine_alignment"
BENCH_ROOT = ROOT / "runs/partial_benchmark"
OUT_DIR = ROOT / "runs/cosine_alignment"


def load_protein_rmsd() -> dict[str, float]:
    """Return name -> Cα RMSD by scanning each *_analysis dir."""
    out: dict[str, float] = {}
    for d in BENCH_ROOT.glob("*_analysis"):
        name = d.name.replace("_analysis", "")
        m = d / "esmfold_direct_eval" / "metrics.json"
        if m.exists():
            out[name] = float(json.loads(m.read_text())["ca_rmsd_after_kabsch"])
    # 7ACN lives elsewhere
    m7 = ROOT / "runs/remote/esmfold_probe_7acn_47layers_consistency1/analysis/esmfold_direct_eval/metrics.json"
    if m7.exists():
        out["7ACN"] = float(json.loads(m7.read_text())["ca_rmsd_after_kabsch"])
    return out


def load_csv(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list] = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def main() -> None:
    rmsds = load_protein_rmsd()
    csv_files = sorted(COS_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no CSVs in {COS_DIR}")

    data = {p.stem: load_csv(p) for p in csv_files}

    # Order proteins by RMSD so the legend reads success -> failure
    names = sorted(data.keys(), key=lambda n: rmsds.get(n, 1e9))

    # Color: green for success (RMSD < 5), red for failure (>= 15), amber middle
    def color(name):
        r = rmsds.get(name, 0)
        if r < 5: return "#2e7d32"
        if r >= 15: return "#c62828"
        return "#ef6c00"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    titles = {
        "cos_neg_grad_ent": "vs −∇(entropy of probe distogram)\n  (loss: model self-confidence)",
        "cos_neg_grad_gt":  "vs −∇(CE against GROUND TRUTH distances)\n  (loss: truth-aligned)",
        "cos_neg_grad_fin": "vs −∇(CE against LAYER-47 probe distogram)\n  (loss: self-prediction)",
        "cos_random":       "vs random direction (baseline)",
    }
    keys = ["cos_neg_grad_ent", "cos_neg_grad_gt", "cos_neg_grad_fin", "cos_random"]

    for ax, key in zip(axes.flatten(), keys):
        for name in names:
            d = data[name]
            ax.plot(d["layer"], d[key], color=color(name), alpha=0.55, lw=1.1,
                    label=f"{name} (RMSD {rmsds.get(name, np.nan):.1f} Å)")
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_xlabel("ESMFold trunk block t")
        ax.set_ylabel("cos(Δz_t, target direction)")
        ax.set_title(titles[key])
        ax.grid(alpha=0.3)
    axes.flatten()[0].legend(fontsize=6.5, loc="upper left", ncol=2)

    fig.suptitle("Per-layer cosine alignment: does ESMFold's trunk update like −∇L for some L?", fontsize=13)
    plt.tight_layout()
    p = OUT_DIR / "cosine_alignment_all.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")

    # ── Summary figure: median cosine over layers 1..46, per protein, per loss ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    rows = []
    for name in names:
        d = data[name]
        rows.append({
            "name": name,
            "rmsd": rmsds.get(name, np.nan),
            "med_cos_ent":  float(np.median(d["cos_neg_grad_ent"])),
            "med_cos_gt":   float(np.median(d["cos_neg_grad_gt"])),
            "med_cos_fin":  float(np.median(d["cos_neg_grad_fin"])),
            "med_cos_rand": float(np.median(d["cos_random"])),
            "mean_late_cos_gt":  float(np.mean(d["cos_neg_grad_gt"][d["layer"] >= 35])),
            "mean_late_cos_fin": float(np.mean(d["cos_neg_grad_fin"][d["layer"] >= 35])),
            "mean_late_cos_ent": float(np.mean(d["cos_neg_grad_ent"][d["layer"] >= 35])),
        })

    # Plot median alignment vs RMSD for each loss
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    keys2 = [
        ("med_cos_ent", "entropy gradient"),
        ("med_cos_gt",  "ground-truth gradient"),
        ("med_cos_fin", "layer-47 gradient"),
    ]
    for ax, (k, lab) in zip(axes, keys2):
        xs = [r["rmsd"] for r in rows]
        ys = [r[k] for r in rows]
        cols = [color(r["name"]) for r in rows]
        ax.scatter(xs, ys, c=cols, s=110, edgecolors="black", lw=0.6)
        for r in rows:
            ax.annotate(r["name"], (r["rmsd"], r[k]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xlabel("Cα RMSD (Å, symlog)")
        ax.set_ylabel(f"median cos(Δz, −∇{lab})")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUT_DIR / "cosine_alignment_vs_rmsd.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"wrote {p}")

    # Save summary CSV
    sc = OUT_DIR / "cosine_alignment_summary.csv"
    with sc.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {sc}")

    # Print summary stats
    succ = [r for r in rows if r["rmsd"] < 5]
    fail = [r for r in rows if r["rmsd"] >= 15]
    print(f"\nSummary: {len(succ)} successes (RMSD<5), {len(fail)} failures (RMSD>=15)")
    for k in ["med_cos_ent", "med_cos_gt", "med_cos_fin", "med_cos_rand"]:
        s_med = np.median([r[k] for r in succ])
        f_med = np.median([r[k] for r in fail])
        print(f"  {k:>18}: success median = {s_med:+.4f}   failure median = {f_med:+.4f}   ratio = {s_med / (abs(f_med) + 1e-9):+.2f}")


if __name__ == "__main__":
    main()
