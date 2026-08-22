#!/usr/bin/env python3
"""Try linear combinations of convexity metrics to maximize correlation with
TM-score. Reports best individual, best pair, best triple."""
from __future__ import annotations
import argparse, csv, itertools, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def style():
    mpl.rcParams.update({
        "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def load(p: Path):
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        for k, v in r.items():
            if k not in ("name", "group"):
                try: r[k] = float(v)
                except: r[k] = float("nan")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    style()

    rows = load(args.csv)
    rows = [r for r in rows if not np.isnan(r.get("tm", float("nan")))]
    n = len(rows)
    tm = np.array([r["tm"] for r in rows])

    candidate_metrics = [
        "ttc98", "ttc99", "log_late_amp", "log_neg_step", "log_auc_late",
        "log_auc_deficit", "log_max_drawdown", "log_d2_std", "log_deficit_area",
        "max_drawdown", "late_amp",
    ]

    def get(m): return np.array([r[m] for r in rows])

    def pearson(x, y):
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12: return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # ── 1. Individual baseline ───────────────────────────────────────────────
    indiv = {m: pearson(get(m), tm) for m in candidate_metrics}
    plddt_r = pearson(get("plddt"), tm)
    print(f"baseline pLDDT r = {plddt_r:+.3f}")
    print(f"best single:  {max(indiv, key=lambda m: abs(indiv[m]))} r = {max(indiv.values(), key=abs):+.3f}")

    # ── 2. All pairs (linear combination via OLS, no intercept needed) ───────
    def best_lin(metrics):
        """Fit TM ~ a₁ x₁ + a₂ x₂ + ... + const. Return (r, coefs, names)."""
        X = np.column_stack([get(m) for m in metrics])
        # Centre
        X_c = X - X.mean(0)
        y_c = tm - tm.mean()
        # OLS
        coef, *_ = np.linalg.lstsq(X_c, y_c, rcond=None)
        pred = X_c @ coef
        r = pearson(pred, y_c)
        return r, coef

    print("\n=== best pairs ===")
    pair_results = []
    for a, b in itertools.combinations(candidate_metrics, 2):
        r, coef = best_lin([a, b])
        pair_results.append(((a, b), r, coef))
    pair_results.sort(key=lambda x: -abs(x[1]))
    for (a, b), r, coef in pair_results[:8]:
        print(f"  {a:>16} + {b:<16}  r={r:+.3f}  coef=({coef[0]:+.3f},{coef[1]:+.3f})")

    print("\n=== best triples ===")
    tri_results = []
    for a, b, c in itertools.combinations(candidate_metrics, 3):
        r, coef = best_lin([a, b, c])
        tri_results.append(((a, b, c), r, coef))
    tri_results.sort(key=lambda x: -abs(x[1]))
    for (a, b, c), r, coef in tri_results[:5]:
        print(f"  {a:>14} + {b:<14} + {c:<14}  r={r:+.3f}")

    # ── 3. Plot: best single, best pair, best triple, pLDDT — 4 panels ───────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.7))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.85, bottom=0.12, wspace=0.26)
    fig.suptitle("Combined convexity metrics vs TM-score  (n=%d)" % n,
                 x=0.04, y=0.95, ha="left", fontsize=22, fontweight="bold")

    plots = [
        ("pLDDT", get("plddt"), plddt_r, False),
    ]
    best_single = max(indiv, key=lambda m: abs(indiv[m]))
    plots.append((best_single, get(best_single), indiv[best_single], False))
    (pa, pb), pr, pcoef = pair_results[0]
    pair_x = pcoef[0] * (get(pa) - get(pa).mean()) + pcoef[1] * (get(pb) - get(pb).mean())
    plots.append((f"{pa} + {pb}", pair_x, pr, True))
    (ta, tb, tc), tr, tcoef = tri_results[0]
    tri_x = sum(tcoef[i] * (get(m) - get(m).mean()) for i, m in enumerate([ta, tb, tc]))
    plots.append((f"{ta}+{tb}+{tc}", tri_x, tr, True))

    for ax, (lab, x, r, centered) in zip(axes, plots):
        for r_, color, marker in [("correct", "#2f7d5a", "o"), ("miss", "#d74b2f", "^")]:
            mask = np.array([row["group"] == r_ for row in rows])
            ax.scatter(x[mask], tm[mask], c=color, marker=marker, s=120,
                       edgecolors="#172033", lw=0.6, label=f"{r_} (n={int(mask.sum())})")
            for row, xv in zip([rr for rr in rows], x):
                if row["group"] == r_:
                    ax.annotate(row["name"], (xv, row["tm"]),
                                fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0.5, color="#0f172a", lw=0.7, ls="--", alpha=0.5)
        ax.text(0.04, 0.96, f"r = {r:+.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=13, fontweight="bold")
        ax.set_xlabel(lab + (" (centered combo)" if centered else ""))
        ax.set_ylabel("TM-score vs PDB"); ax.grid(True, lw=0.6)
        ax.set_title(lab, fontsize=12)
    axes[0].legend(loc="lower right", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
