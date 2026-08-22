#!/usr/bin/env python3
"""
Standard convergence diagnostics applied to ESMFold's trunk trajectories.

Rather than inventing ad-hoc metrics (tortuosity, late-step velocity), we apply
classical time-series stationarity tests and MCMC convergence diagnostics:

  • ADF  (Augmented Dickey-Fuller) — null: has unit root (non-stationary/drift).
                                     Reject → stationary → converged.
  • KPSS                            — null: stationary.
                                     Reject → non-stationary → drift.
  • Geweke                          — compares first 10% to last 50% of the chain.
                                     |z| > 2 → not converged.
  • IAT (integrated autocorrelation time) — 1 + 2*Σρ(k). Small → mixed/converged.
                                            Large → drifting (everything correlated).

Each test is applied to a 1-D summary of the trunk trajectory per block. We use
the distance-to-centroid summary (already tested visually), but the same tests
work on any per-block scalar.

For a successfully-folded protein, the trunk's trajectory is *stationary* (the
optimizer has reached a fixed point and oscillates near it). For a failure case,
it is *non-stationary* (continuous descent / drift). Standard diagnostics should
cleanly distinguish these regimes.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

PROTEINS = [
    ("1AO6 (albumin, 578)",
     "runs/remote/esmfold_probe_1ao6_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
     "success"),
    ("7ACN (aconitase, 753)",
     "runs/remote/esmfold_probe_7acn_47layers_consistency1/analysis/mds_structure_video_fixed_chunked/distogram_mds_models.pdb",
     "success"),
    ("8FMI (KRas, 170)",
     "runs/remote/esmfold_probe_8fmi_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
     "success"),
    ("T1137s6 (CASP fail)",
     "runs/remote/esmfold_probe_T1137s6_47layers_consistency1/analysis/mds_structure_video_fixed/distogram_mds_models.pdb",
     "failure"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_models(path: Path) -> np.ndarray:
    """Return (n_blocks, n_residues, 3) — Cα coords from each MODEL in PDB."""
    models, current = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):    current = []
            elif line.startswith("ENDMDL"): models.append(current); current = None
            elif line.startswith("ATOM") and current is not None:
                current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(models)


# ──────────────────────────────────────────────────────────────────────────────
# 1-D trajectory summaries
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_distances(c: np.ndarray) -> np.ndarray:
    d = c[:, None, :] - c[None, :, :]
    return np.sqrt((d**2).sum(-1))


def step_size_distogram(coords: np.ndarray) -> np.ndarray:
    """
    Mean absolute change in pairwise distances between consecutive blocks.
    For a converged trajectory, this should be small and stationary.
    For a drifting trajectory, this should be larger AND have a downward trend
    (the gradient is shrinking but hasn't reached zero).
    """
    R = coords.shape[1]
    iu = np.triu_indices(R, k=1)
    D = np.stack([pairwise_distances(c)[iu] for c in coords])
    return np.abs(D[1:] - D[:-1]).mean(axis=1)   # (L-1,)


def distance_corr_vs_centroid_distogram(coords: np.ndarray) -> np.ndarray:
    """
    Per-block: correlation of this block's distance matrix with the *centroid*
    distogram (mean over all blocks). For orbiting trajectory, this is high and
    stationary. For drift, the centroid is meaningless and the correlation
    follows the U-shape we saw qualitatively.
    """
    R = coords.shape[1]
    iu = np.triu_indices(R, k=1)
    D = np.stack([pairwise_distances(c)[iu] for c in coords])
    Dmean = D.mean(axis=0)
    return np.array([float(np.corrcoef(D[t], Dmean)[0, 1]) for t in range(len(D))])


# ──────────────────────────────────────────────────────────────────────────────
# Convergence diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def integrated_autocorrelation_time(x: np.ndarray, c: int = 5) -> float:
    """
    IAT = 1 + 2 Σ_{k=1}^{M} ρ(k), with automated cutoff M = c·IAT_estimate.

    Following Sokal's "automated windowing" heuristic.
    """
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    # autocorrelation via FFT
    f = np.fft.fft(x, n=2*n)
    acf = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    acf /= acf[0]

    tau = 1.0
    for M in range(1, n):
        tau_new = 1.0 + 2.0 * acf[1:M+1].sum()
        if M >= c * tau_new:
            return float(tau_new)
        tau = tau_new
    return float(tau)


def geweke_z(x: np.ndarray, first: float = 0.10, last: float = 0.50) -> float:
    """
    Geweke convergence diagnostic.

    Compare mean of first `first` fraction of chain to last `last` fraction.
    Returns z-score; |z| > 2 typically indicates non-convergence.
    """
    x = np.asarray(x)
    n = len(x)
    n1 = max(2, int(n * first))
    n2 = max(2, int(n * last))
    seg1 = x[:n1]
    seg2 = x[-n2:]

    def spectral_var(s):
        # variance of mean estimator, accounting for autocorrelation
        s = s - s.mean()
        nn = len(s)
        f = np.fft.fft(s, n=2*nn)
        acf = np.real(np.fft.ifft(f * np.conj(f)))[:nn] / nn
        # IAT-style truncation
        tau = 1.0
        for M in range(1, nn):
            new = 1.0 + 2.0 * (acf[1:M+1] / acf[0]).sum() if acf[0] > 1e-12 else 1.0
            if M >= 5 * new:
                tau = new
                break
            tau = new
        return float(np.var(s) * tau / nn)

    v1 = spectral_var(seg1)
    v2 = spectral_var(seg2)
    denom = (v1 + v2) ** 0.5
    if denom < 1e-12:
        return 0.0
    return float((seg1.mean() - seg2.mean()) / denom)


def mann_kendall_trend(x: np.ndarray) -> tuple[float, float]:
    """
    Non-parametric Mann-Kendall test for monotone trend.
    Null: no trend.  Reject → monotone trend (drift).
    Returns (z-statistic, two-sided p-value).
    """
    from scipy.stats import norm
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i+1:] - x[i]).sum()
    # variance under null (no ties)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def linear_trend_slope(x: np.ndarray) -> tuple[float, float]:
    """OLS slope vs time index, plus p-value of slope ≠ 0."""
    from scipy.stats import linregress
    t = np.arange(len(x))
    res = linregress(t, x)
    return float(res.slope), float(res.pvalue)


def run_diagnostics(x: np.ndarray) -> dict:
    """Apply diagnostics to a 1-D trajectory."""
    x = np.asarray(x, dtype=float)

    try:
        adf_stat, adf_p, *_ = adfuller(x, regression="c", autolag="AIC")
    except Exception:
        adf_stat, adf_p = float("nan"), float("nan")

    try:
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = float("nan"), float("nan")

    iat   = integrated_autocorrelation_time(x)
    n_eff = len(x) / max(iat, 1e-6)
    gz    = geweke_z(x)
    mk_z, mk_p   = mann_kendall_trend(x)
    slope, sp_p  = linear_trend_slope(x)

    return {
        "n":         len(x),
        "mean":      float(x.mean()),
        "std":       float(x.std()),
        "adf_p":     adf_p,
        "kpss_p":    kpss_p,
        "iat":       iat,
        "n_eff":     n_eff,
        "geweke_z":  gz,
        "mk_z":      mk_z,
        "mk_p":      mk_p,
        "slope":     slope,
        "slope_p":   sp_p,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    summaries: dict[str, dict] = {}

    for label, lw_path, status in PROTEINS:
        path = ROOT / lw_path
        if not path.exists():
            print(f"[skip] {label}: missing {lw_path}")
            continue
        coords = parse_models(path)            # (47, R, 3)

        # Step size between consecutive blocks: should be stationary if converged
        x_step = step_size_distogram(coords)
        d1 = run_diagnostics(x_step)
        summaries[label] = {"status": status, "step": d1}

        print(f"\n══ {label}  ({status})  ══")
        print(f"  trajectory summary = step size between consecutive blocks (Å per pair)")
        print(f"    mean = {d1['mean']:.4f}   std = {d1['std']:.4f}")
        print(f"    ADF       p = {d1['adf_p']:.4f}  (< 0.05 → stationary, no unit root)")
        print(f"    KPSS      p = {d1['kpss_p']:.4f}  (> 0.05 → stationary)")
        print(f"    Mann-Kendall trend z = {d1['mk_z']:+6.2f}   p = {d1['mk_p']:.4f}  "
              f"(< 0.05 → monotone trend)")
        print(f"    OLS slope  = {d1['slope']:+8.5f}   p = {d1['slope_p']:.4f}")
        print(f"    IAT       = {d1['iat']:6.2f}   N_eff = {d1['n_eff']:.1f}/{d1['n']}")
        print(f"    Geweke z  = {d1['geweke_z']:+6.3f}  (|z|<2 → converged)")

    # ── Side-by-side summary table ────────────────────────────────────────────
    print("\n" + "═" * 110)
    print("SUMMARY · trajectory = step size between consecutive blocks (Å per pair)")
    print("═" * 110)
    print(f"{'protein':>24}  {'truth':>7}  {'mean':>7}  {'ADF p':>7}  {'KPSS p':>7}  "
          f"{'MK p':>7}  {'slope':>10}  {'|Geweke|':>9}  verdict")
    for label, s in summaries.items():
        d = s["step"]
        # Two evidence types:
        #   (A) is the series stationary?  ADF rejects unit root + KPSS fails to reject stationarity
        #   (B) is there a monotone trend? Mann-Kendall + linear slope
        stat_evidence  = (d['adf_p'] < 0.05) + (d['kpss_p'] > 0.05)
        trend_evidence = (d['mk_p']  < 0.05) + (d['slope_p'] < 0.05)

        if trend_evidence >= 1:
            verdict = "TREND (drift)"
        elif stat_evidence >= 1:
            verdict = "stationary (converged)"
        else:
            verdict = "ambiguous"

        print(f"{label:>24}  {s['status']:>7}  "
              f"{d['mean']:7.4f}  {d['adf_p']:7.4f}  {d['kpss_p']:7.4f}  "
              f"{d['mk_p']:7.4f}  {d['slope']:+10.5f}  "
              f"{abs(d['geweke_z']):9.3f}  {verdict}")

    print("\nInterpretation:")
    print("  ADF p < 0.05            → rejects unit root → stationary (converged)")
    print("  KPSS p > 0.05           → fails to reject stationarity → consistent with convergence")
    print("  Mann-Kendall p < 0.05   → significant monotone trend → drift, NOT converged")
    print("  OLS slope significantly nonzero → linear drift, NOT converged")
    print("  |Geweke z| < 2          → first 10% and last 50% have same mean → converged")


if __name__ == "__main__":
    main()
