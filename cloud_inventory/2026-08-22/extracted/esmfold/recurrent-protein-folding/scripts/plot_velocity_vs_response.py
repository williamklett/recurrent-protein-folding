#!/usr/bin/env python3
"""Test the convexity hypothesis: does the layer-20 response track the
LOCAL z-velocity at layer 20 for each protein?

  - Plot ||z_{t+1} − z_t|| vs layer for each protein
  - Highlight layer 20 (injection point)
  - Overlay each protein's measured global RMSD response
  - Test: is response correlated with velocity at the injection layer?
"""
from __future__ import annotations

import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ATTR = ROOT / "runs/attractor"
PARTIAL = ROOT / "runs/partial_benchmark"


def load_rmsd():
    out = {}
    for d in PARTIAL.glob("*_analysis"):
        m = d / "esmfold_direct_eval/metrics.json"
        if m.exists():
            out[d.name.replace("_analysis", "")] = json.loads(m.read_text())["ca_rmsd_after_kabsch"]
    m7 = ROOT / "runs/remote/esmfold_probe_7acn_47layers_consistency1/analysis/esmfold_direct_eval/metrics.json"
    if m7.exists(): out["7ACN"] = json.loads(m7.read_text())["ca_rmsd_after_kabsch"]
    return out


def main():
    # Load velocity curves
    vel = {}
    with (ATTR / "z_velocity.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            n = row["name"]
            vel.setdefault(n, []).append((int(row["layer"]), float(row["z_velocity"])))
    for n in vel:
        vel[n].sort()

    # Load attractor responses
    resp = {}
    for d in ATTR.iterdir():
        sj = d / "summary.json"
        if not sj.exists(): continue
        s = json.loads(sj.read_text())
        resp[s["name"]] = {
            "global_rmsd": s["mean_global_rmsd"],
            "delta_pair": s["mean_abs_delta_pair"],
            "ratio": s["median_response_ratio"],
        }

    rmsds = load_rmsd()

    proteins = sorted(set(vel) & set(resp), key=lambda n: rmsds.get(n, 999))
    def color(name):
        r = rmsds.get(name, 0)
        if r < 5: return "#2e7d32"
        if r >= 15: return "#c62828"
        return "#ef6c00"

    # ── Plot 1: velocity curves with layer-20 marker ──────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for name in proteins:
        layers = [l for l, _ in vel[name]]
        vels = [v for _, v in vel[name]]
        ax.plot(layers, vels, color=color(name), alpha=0.7, lw=1.5,
                label=f"{name} (RMSD {rmsds.get(name, np.nan):.1f}Å, resp={resp[name]['global_rmsd']:.3f}Å)")
        # Mark injection layer
        for l, v in vel[name]:
            if l == 20:
                ax.scatter([20], [v], color=color(name), s=150, edgecolors="black", lw=1.5, zorder=10)
    ax.axvline(20, color="gray", ls="--", lw=1, alpha=0.7, label="injection layer = 20")
    ax.set_xlabel("ESMFold trunk block t")
    ax.set_ylabel("||z_{t+1} − z_t||_F  (raw z-step magnitude)")
    ax.set_title("z-velocity curve for each protein.  Black-edged dots = velocity at injection layer 20")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = ATTR / "velocity_curves.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"wrote {p}")

    # ── Plot 2: response vs velocity-at-injection-layer ───────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    rows = []
    for name in proteins:
        v20 = next((v for l, v in vel[name] if l == 20), None)
        if v20 is None: continue
        rmsd_r = resp[name]["global_rmsd"]
        ca_rmsd = rmsds.get(name, np.nan)
        rows.append((name, v20, rmsd_r, ca_rmsd))
        ax.scatter(v20, rmsd_r, s=200, color=color(name), edgecolors="black", lw=1)
        ax.annotate(f"{name}\nRMSD {ca_rmsd:.1f}Å", (v20, rmsd_r),
                    fontsize=8, xytext=(8, 8), textcoords="offset points")

    # Pearson r
    v_arr = np.array([r[1] for r in rows])
    resp_arr = np.array([r[2] for r in rows])
    r_pearson = float(np.corrcoef(v_arr, resp_arr)[0, 1])

    ax.set_xlabel("z-velocity at injection layer 20  (Frobenius norm)")
    ax.set_ylabel("Global RMSD response to +5Å distogram nudge  (Å)")
    ax.set_title(f"Does layer-20 velocity predict response?  Pearson r = {r_pearson:+.3f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = ATTR / "response_vs_velocity_at_inject.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"wrote {p}")
    print(f"Pearson r(velocity_at_20, global_response) = {r_pearson:+.3f}")

    # Print the data
    print(f"\n{'protein':>10} {'CA RMSD':>9} {'vel@20':>10} {'response':>10}")
    for name, v20, rmsd_r, ca_rmsd in sorted(rows, key=lambda x: x[1]):
        print(f"{name:>10} {ca_rmsd:>8.2f}Å {v20:>9.1f}  {rmsd_r:>8.4f}Å")


if __name__ == "__main__":
    main()
