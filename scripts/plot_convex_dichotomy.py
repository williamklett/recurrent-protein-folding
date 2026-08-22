#!/usr/bin/env python3
"""Final figure for the basin-convexity result.

Three panels:
  A. Distogram velocity curve per protein, colored by class (convex vs non-convex)
  B. Layer-20 RMSD response vs peak-layer RMSD response (does it change?)
  C. Peak-velocity layer vs RMSD response (the discriminator)
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ATTR_PEAK = ROOT / "runs/attractor_peak"
ATTR_L20  = ROOT / "runs/attractor"
PARTIAL   = ROOT / "runs/partial_benchmark"


def load_rmsd():
    out = {}
    for d in PARTIAL.glob("*_analysis"):
        m = d / "esmfold_direct_eval/metrics.json"
        if m.exists():
            out[d.name.replace("_analysis", "")] = json.loads(m.read_text())["ca_rmsd_after_kabsch"]
    return out


def load_summaries(root: Path) -> dict[str, dict]:
    out = {}
    for d in root.iterdir():
        sj = d / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            out[s["name"]] = s
    return out


def main():
    rmsds = load_rmsd()
    peak  = load_summaries(ATTR_PEAK)
    l20   = load_summaries(ATTR_L20)

    # Build a table
    rows = []
    for name in sorted(peak.keys()):
        if name not in l20: continue
        rows.append({
            "name": name,
            "ca_rmsd": rmsds.get(name, np.nan),
            "peak_layer": peak[name]["inject_layer"],
            "peak_rmsd_resp": peak[name]["mean_global_rmsd"],
            "peak_dpair": peak[name]["mean_abs_delta_pair"],
            "l20_rmsd_resp": l20[name]["mean_global_rmsd"],
            "l20_dpair": l20[name]["mean_abs_delta_pair"],
        })

    # Classify: non-convex = global response > 0.05 Å at peak
    THRESHOLD = 0.05
    for r in rows:
        r["non_convex"] = r["peak_rmsd_resp"] > THRESHOLD

    # Load distogram velocity curves
    vel = {}
    with (ATTR_L20 / "dist_velocity.csv").open() as f:
        for row in csv.DictReader(f):
            vel.setdefault(row["name"], []).append((int(row["layer"]), float(row["dist_mae"])))
    for n in vel: vel[n].sort()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── Panel A: velocity curves ─────────────────────────────────────────────
    ax = axes[0]
    for r in rows:
        name = r["name"]
        if name not in vel: continue
        pts = vel[name]
        ls = pts[:46]  # drop the layer-47 commit burst that dominates everything
        layers = [p[0] for p in ls]
        velocities = [p[1] for p in ls]
        color = "#c62828" if r["non_convex"] else "#2e7d32"
        ax.plot(layers, velocities, color=color, alpha=0.7, lw=1.5,
                label=f"{name} (RMSD {r['ca_rmsd']:.1f}Å)")
        # Mark peak
        ax.scatter([r["peak_layer"]], [vel[name][r["peak_layer"] - 1][1]],
                   color=color, s=120, edgecolors="black", lw=1.2, zorder=10)
    ax.set_xlabel("ESMFold trunk block t  (peak at black dot)")
    ax.set_ylabel("Distogram velocity: mean |D_{t+1} − D_t|  (Å)")
    ax.set_title("Per-protein mid-trunk distogram velocity\n(red = non-convex, green = convex)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)

    # ── Panel B: response at peak vs response at L=20 ────────────────────────
    ax = axes[1]
    for r in rows:
        color = "#c62828" if r["non_convex"] else "#2e7d32"
        ax.scatter([r["l20_rmsd_resp"]], [r["peak_rmsd_resp"]],
                   color=color, s=180, edgecolors="black", lw=1)
        ax.annotate(r["name"], (r["l20_rmsd_resp"], r["peak_rmsd_resp"]),
                    fontsize=8, xytext=(6, 6), textcoords="offset points")
    lim = max(0.25, max(r["peak_rmsd_resp"] for r in rows) * 1.1)
    ax.plot([0, lim], [0, lim], color="gray", ls="--", lw=0.7, label="y = x (no layer dependence)")
    ax.set_xlabel("Global RMSD response @ L=20  (Å)")
    ax.set_ylabel("Global RMSD response @ peak layer  (Å)")
    ax.set_title("Convex basins: response is layer-independent")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel C: peak velocity layer vs peak response ────────────────────────
    ax = axes[2]
    for r in rows:
        color = "#c62828" if r["non_convex"] else "#2e7d32"
        ax.scatter([r["peak_layer"]], [r["peak_rmsd_resp"]],
                   color=color, s=180, edgecolors="black", lw=1)
        ax.annotate(r["name"], (r["peak_layer"], r["peak_rmsd_resp"]),
                    fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax.axhline(THRESHOLD, color="gray", ls="--", lw=0.7, label=f"threshold = {THRESHOLD} Å")
    ax.axvspan(25, 32, color="orange", alpha=0.12, label="mid-trunk uncertainty band")
    ax.set_xlabel("Peak distogram-velocity layer")
    ax.set_ylabel("Global RMSD response @ peak  (Å)")
    ax.set_title("Non-convexity localized to mid-trunk peaks (L≈27-29)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = ATTR_PEAK / "convex_dichotomy.png"
    plt.savefig(p, dpi=160); plt.close()
    print(f"wrote {p}")

    # Print summary
    print(f"\nResult: {sum(1 for r in rows if r['non_convex'])} non-convex / {len(rows)} total")
    print(f"\n{'name':>10} {'RMSD':>7} {'peak_L':>7} {'resp@peak':>11} {'resp@L20':>11} {'class':>14}")
    for r in sorted(rows, key=lambda x: x["peak_rmsd_resp"]):
        cls = "NON-CONVEX" if r["non_convex"] else "convex"
        print(f"{r['name']:>10} {r['ca_rmsd']:>6.1f}Å {r['peak_layer']:>7d} {r['peak_rmsd_resp']:>10.4f}Å {r['l20_rmsd_resp']:>10.4f}Å {cls:>14}")


if __name__ == "__main__":
    main()
