#!/usr/bin/env python3
"""
Local breathing/drift analysis.

Experiments 1-4:
  1. Tortuosity & oscillation_score on Albumin (1AO6) vs T1137s6
  2. Per-residue breathing amplitude vs experimental B-factors (Albumin)
  3. Autocorrelation of layer-to-layer step direction
  4. Publication-quality figures

Inputs: layerwise Cα coordinates extracted from
    runs/remote/esmfold_probe_<protein>_47layers_consistency1/analysis/
        mds_structure_video_fixed/distogram_mds_models.pdb

Each PDB contains 47 MODELs (one per ESMFold trunk block), with per-residue Cα
coords reconstructed from the probe's distogram via MDS embedding.
"""
from __future__ import annotations

import sys
from pathlib import Path
import urllib.request

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent

PROTEINS = {
    "Albumin (1AO6)": ROOT / "runs/remote/esmfold_probe_1ao6_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
    "T1137s6":        ROOT / "runs/remote/esmfold_probe_T1137s6_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
}


# ──────────────────────────────────────────────────────────────────────────────
# PDB parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_models(pdb_path: Path) -> np.ndarray:
    """Return Cα coords with shape (n_models, n_residues, 3)."""
    models: list[list[np.ndarray]] = []
    current: list[np.ndarray] | None = None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ENDMDL"):
                models.append(current)
                current = None
            elif line.startswith("ATOM") and current is not None:
                # Cα-only PDB, atoms already filtered
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                current.append(np.array([x, y, z]))
    return np.stack([np.stack(m) for m in models])   # (L, R, 3)


def fetch_pdb_bfactors(pdb_id: str, chain: str = "A") -> tuple[np.ndarray, list[int]]:
    """Download a PDB from RCSB and return per-residue Cα B-factors (chain A)."""
    cache = ROOT / "data" / f"{pdb_id}.pdb"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, cache)
    b: dict[int, float] = {}
    with open(cache) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            ch   = line[21]
            if atom != "CA" or ch != chain:
                continue
            res_seq = int(line[22:26])
            bfac    = float(line[60:66])
            if res_seq not in b:   # first-altloc only
                b[res_seq] = bfac
    res_ids = sorted(b)
    return np.array([b[r] for r in res_ids]), res_ids


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory metrics
# ──────────────────────────────────────────────────────────────────────────────

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Align P onto Q by least-squares rotation; return aligned P."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return Pc @ R.T + Q.mean(0)


def align_trajectory(coords: np.ndarray) -> np.ndarray:
    """Align every frame onto frame 0 (Kabsch).  Input/output shape: (L, R, 3)."""
    ref = coords[0]
    out = [ref]
    for f in coords[1:]:
        out.append(kabsch_align(f, ref))
    return np.stack(out)


def ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def trajectory_metrics(coords_aligned: np.ndarray) -> dict:
    """
    coords_aligned: (L, R, 3), already Kabsch-aligned to the first frame.
    Returns: tortuosity, oscillation_score, mean/std dist to centroid,
             path_length, end_to_end, plus per-layer step lengths.
    """
    L = coords_aligned.shape[0]
    centroid = coords_aligned.mean(axis=0)               # (R, 3) — average structure

    # Distance of each frame from centroid (RMSD to mean structure)
    d_to_centroid = np.array([ca_rmsd(f, centroid) for f in coords_aligned])  # (L,)

    # Step lengths (RMSD between consecutive frames)
    steps = np.array([ca_rmsd(coords_aligned[t+1], coords_aligned[t])
                      for t in range(L-1)])              # (L-1,)

    # End-to-end distance (frame 0 vs final frame)
    end_to_end = ca_rmsd(coords_aligned[0], coords_aligned[-1])

    path_length = steps.sum()
    tortuosity  = path_length / max(end_to_end, 1e-8)

    osc_score   = d_to_centroid.std() / max(d_to_centroid.mean(), 1e-8)
    # ↑ LOW = stable orbit (all frames equidistant from center)
    #   HIGH = drifting trajectory (centroid is meaningless midpoint, big spread)

    return {
        "path_length":   path_length,
        "end_to_end":    end_to_end,
        "tortuosity":    tortuosity,
        "mean_d_centroid": d_to_centroid.mean(),
        "std_d_centroid":  d_to_centroid.std(),
        "osc_score":     osc_score,
        "d_to_centroid": d_to_centroid,
        "steps":         steps,
        "centroid":      centroid,
    }


def step_autocorrelation(coords_aligned: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Sign-correlation of consecutive step vectors at various lags.
    Returns array (max_lag,) of mean cosine between Δ_t and Δ_{t+k}.

    For oscillating trajectory: should show sign-flip pattern.
    For drifting trajectory: monotone decay from +1.
    """
    L = coords_aligned.shape[0]
    # Flatten residue+coord dims so each step is a single vector
    flat = coords_aligned.reshape(L, -1)                  # (L, R*3)
    deltas = flat[1:] - flat[:-1]                         # (L-1, R*3)
    # Normalize
    deltas /= (np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-8)
    acorr = []
    for k in range(1, max_lag + 1):
        if k >= deltas.shape[0]:
            acorr.append(np.nan)
            continue
        # mean inner product of step t with step t+k
        ips = (deltas[:-k] * deltas[k:]).sum(axis=1)
        acorr.append(ips.mean())
    return np.array(acorr)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 78)
    print("EXPERIMENT 1 + 3: Trajectory metrics & autocorrelation")
    print("=" * 78)

    trajectories: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    autocorrs: dict[str, np.ndarray] = {}

    for name, path in PROTEINS.items():
        if not path.exists():
            print(f"  [skip] {name}: PDB not found at {path}")
            continue
        coords = parse_models(path)                       # (47, R, 3)
        coords = align_trajectory(coords)
        trajectories[name] = coords

        m = trajectory_metrics(coords)
        ac = step_autocorrelation(coords, max_lag=20)
        metrics[name]   = m
        autocorrs[name] = ac

        print(f"\n{name}: shape={coords.shape}")
        print(f"  path length    = {m['path_length']:8.3f} Å")
        print(f"  end-to-end     = {m['end_to_end']:8.3f} Å")
        print(f"  tortuosity     = {m['tortuosity']:8.3f}        "
              f"(≈1 = drift, >>1 = orbit)")
        print(f"  ⟨d_centroid⟩   = {m['mean_d_centroid']:8.3f} Å")
        print(f"  σ(d_centroid)  = {m['std_d_centroid']:8.3f} Å")
        print(f"  osc_score      = {m['osc_score']:8.3f}        "
              f"(LOW = stable orbit, HIGH = drift)")
        print(f"  step lengths:  min={m['steps'].min():.3f}  "
              f"mean={m['steps'].mean():.3f}  max={m['steps'].max():.3f}")

        # First few autocorr values
        ac_str = "  ".join(f"k={k+1}:{v:+.3f}" for k, v in enumerate(ac[:5]))
        print(f"  step autocorr (first 5 lags): {ac_str}")
        first_neg = next((k+1 for k, v in enumerate(ac) if v < 0), None)
        print(f"  first negative-correlation lag: {first_neg}  "
              f"(early = high-freq oscillation, none = drift)")

    print("\n" + "=" * 78)
    print("EXPERIMENT 2: Per-residue breathing amplitude vs experimental B-factors")
    print("=" * 78)

    bfactor_corr = None
    if "Albumin (1AO6)" in trajectories:
        traj_alb = trajectories["Albumin (1AO6)"]         # (47, R_mod, 3)
        R_mod = traj_alb.shape[1]
        # Per-residue breathing amplitude = std of position across layers
        # (already Kabsch-aligned, so this is just position variance)
        per_res_amp = traj_alb.std(axis=0).mean(axis=1)   # (R_mod,)

        # Get experimental B-factors from real 1AO6 crystal structure
        bfac, res_ids = fetch_pdb_bfactors("1AO6", chain="A")
        print(f"\n1AO6 chain A: {len(bfac)} residues with Cα B-factors")
        print(f"  Model trajectory: {R_mod} residues")

        # Align lengths (truncate to common length from the N-terminus)
        n = min(len(bfac), R_mod)
        amp = per_res_amp[:n]
        bf  = bfac[:n]

        # Correlate
        from numpy import corrcoef
        r = float(corrcoef(amp, bf)[0, 1])
        bfactor_corr = r
        print(f"  Truncated to first {n} residues for comparison")
        print(f"  Pearson r(breathing_amplitude, B_factor) = {r:.3f}")
        if abs(r) > 0.3:
            print("    → significant correlation: model's internal motion mirrors thermal flexibility")
        else:
            print("    → weak correlation: model breathing may not match experimental flexibility")

        # Save side-by-side for the figure
        np.savez(ROOT / "runs/breathing_vs_bfactor_albumin.npz",
                 breathing_amplitude=amp, bfactor=bf, residue_index=np.arange(n))

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 4: Publication figures
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("EXPERIMENT 4: Publication-quality figures")
    print("=" * 78)

    out_dir = ROOT / "runs" / "breathing_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: distance-to-centroid trace (the oscillation signature)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    colors = {"Albumin (1AO6)": "#1976D2", "T1137s6": "#D32F2F"}
    for name, m in metrics.items():
        ax.plot(np.arange(1, len(m["d_to_centroid"]) + 1),
                m["d_to_centroid"], marker="o", lw=1.5, ms=4,
                color=colors.get(name, "k"),
                label=f"{name}  (osc_score={m['osc_score']:.2f}, τ={m['tortuosity']:.2f})")
    ax.set_xlabel("ESMFold trunk block")
    ax.set_ylabel("Distance from layerwise centroid (Å, Cα RMSD)")
    ax.set_title("Trajectory dynamics: bounded orbit vs continuous drift")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_dir / "fig1_distance_to_centroid.png"
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"  Saved {p}")

    # Figure 2: step autocorrelation
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for name, ac in autocorrs.items():
        lags = np.arange(1, len(ac) + 1)
        ax.plot(lags, ac, marker="o", lw=1.5, ms=4,
                color=colors.get(name, "k"), label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Lag k (blocks)")
    ax.set_ylabel("Mean cosine(Δ_t, Δ_{t+k})")
    ax.set_title("Step-direction autocorrelation: oscillation flips sign, drift doesn't")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_dir / "fig2_step_autocorrelation.png"
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"  Saved {p}")

    # Figure 3: per-residue breathing amplitude vs B-factor (Albumin)
    if bfactor_corr is not None:
        d = np.load(ROOT / "runs/breathing_vs_bfactor_albumin.npz")
        amp, bf, idx = d["breathing_amplitude"], d["bfactor"], d["residue_index"]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        axes[0].plot(idx, amp, color="#7B1FA2", lw=1)
        axes[0].set_ylabel("ESMFold breathing amp\n(Å, σ across layers)")
        axes[0].set_title(f"Albumin (1AO6): model breathing vs crystal B-factors  "
                          f"(Pearson r = {bfactor_corr:.3f})")
        axes[0].grid(alpha=0.3)

        axes[1].plot(idx, bf, color="#388E3C", lw=1)
        axes[1].set_xlabel("Residue index")
        axes[1].set_ylabel("Crystal B-factor (Å²)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / "fig3_breathing_vs_bfactor.png"
        plt.savefig(p, dpi=140)
        plt.close()
        print(f"  Saved {p}")

        # Also a scatterplot
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        ax.scatter(amp, bf, s=8, alpha=0.5, color="#7B1FA2")
        ax.set_xlabel("ESMFold breathing amplitude (Å)")
        ax.set_ylabel("Crystal B-factor (Å²)")
        ax.set_title(f"Albumin 1AO6  ·  Pearson r = {bfactor_corr:.3f}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = out_dir / "fig3b_breathing_vs_bfactor_scatter.png"
        plt.savefig(p, dpi=140)
        plt.close()
        print(f"  Saved {p}")

    # Summary table
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"{'protein':>20}  {'tortuosity':>11}  {'osc_score':>10}  "
          f"{'first<0 lag':>12}  {'mean step Å':>12}")
    for name, m in metrics.items():
        ac = autocorrs[name]
        fn = next((k+1 for k, v in enumerate(ac) if v < 0), -1)
        fn_s = str(fn) if fn > 0 else "none"
        print(f"{name:>20}  {m['tortuosity']:11.3f}  {m['osc_score']:10.3f}  "
              f"{fn_s:>12}  {m['steps'].mean():12.3f}")
    if bfactor_corr is not None:
        print(f"\nB-factor correlation (Albumin breathing vs crystal): r = {bfactor_corr:.3f}")


if __name__ == "__main__":
    main()
