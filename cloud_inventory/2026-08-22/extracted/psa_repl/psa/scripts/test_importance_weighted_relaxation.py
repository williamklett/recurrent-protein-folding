"""Test the 'importance-weighted constraint relaxation' hypothesis.

Claim: during the trunk's non-metric phase (blocks ~23–40), per-pair edit
magnitude should ANTI-correlate with the pair's evolutionary / structural
'importance' — important constraints are preserved, unimportant ones are
relaxed.

Three proxies for importance, per pair (i, j):
  1. final_confidence  = −entropy(distogram@block47)[i,j]
                          (the model's own final confidence)
  2. esm2_attention    = mean ESM-2 attention at (i,j) across all 36 layers,
                          all 40 heads, both directions
                          (the LM's learned coupling signal)
  3. is_contact        = ground-truth contact (true binary importance)

For each val protein and each block-transition in the non-metric phase:
  edit_magnitude(i, j) = |pred_dist[b+1] - pred_dist[b]|
Aggregate over block-transitions in [23..40] → total relaxation activity.

Correlate (across all pairs, all proteins) the relaxation activity against
each importance proxy.

Outputs:
  runs/importance_relaxation.json
  runs/importance_relaxation.png
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("importance")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
DISTO_DIR = ROOT / "data" / "distograms"
ESM2_DIR = ROOT / "data" / "esm2_attention"
RUNS_DIR = ROOT / "runs"

NONMETRIC_RANGE = (23, 41)   # block transitions used to define "relaxation activity"
FINAL_BLOCK = 47


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def load_protein_features(pdb_id: str) -> dict | None:
    """Load all per-pair features for one protein, aligned to L."""
    disto_path = DISTO_DIR / f"{pdb_id.lower()}.pt"
    if not disto_path.exists():
        return None
    disto = torch.load(disto_path, map_location="cpu", weights_only=False)
    phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
    L = disto["L"]
    if phys["distances"].shape[0] != L:
        return None
    pred = disto["pred_distance"].float().numpy()       # (48, L, L)
    ent = disto["pred_entropy"].float().numpy()         # (48, L, L)
    # ESM-2 attention is optional (may not be locally cached)
    esm2_path = ESM2_DIR / f"{pdb_id.lower()}.pt"
    attn = None
    if esm2_path.exists():
        esm2 = torch.load(esm2_path, map_location="cpu", weights_only=False)
        attn_t = esm2["attn"].float().numpy()
        if attn_t.shape[3] == L:
            attn = attn_t
    return {"pdb_id": pdb_id, "L": L, "pred": pred, "ent": ent, "attn": attn,
             "gt_dist": phys["distances"].astype(np.float32),
             "contacts": phys["contacts"].astype(bool)}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def main():
    np.random.seed(0); random.seed(0)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=0)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    per_protein = []
    pooled = {"relax_act": [], "final_conf": [], "esm2_attn": [], "is_contact": [],
               "gt_dist": [], "seq_sep": []}

    for pid in val_ids:
        p = load_protein_features(pid)
        if p is None:
            log.warning("skip %s (missing data)", pid)
            continue
        L = p["L"]
        eye = np.eye(L, dtype=bool)
        mask = ~eye

        # Relaxation activity: sum of |Δd| over non-metric phase transitions
        b0, b1 = NONMETRIC_RANGE
        rel_act = np.zeros((L, L), dtype=np.float32)
        for b in range(b0, b1):
            rel_act += np.abs(p["pred"][b + 1] - p["pred"][b])

        # Importance proxies
        final_conf = -p["ent"][FINAL_BLOCK]                    # (L, L)
        is_contact = p["contacts"].astype(np.float32)

        # Per-protein correlations
        ra = rel_act[mask]
        fc = final_conf[mask]
        ct = is_contact[mask]
        row = {
            "pdb_id": pid, "L": L,
            "pearson_relax_vs_finalconf": pearson(ra, fc),
            "spearman_relax_vs_finalconf": spearman(ra, fc),
            "pearson_relax_vs_contact": pearson(ra, ct),
            "spearman_relax_vs_contact": spearman(ra, ct),
            "mean_relax_contact": float(rel_act[p["contacts"] & mask].mean()),
            "mean_relax_noncontact": float(rel_act[(~p["contacts"]) & mask].mean()),
        }
        if p["attn"] is not None:
            esm2_attn = p["attn"][:, 0].mean(axis=(0, 1))      # (L, L)
            ea = esm2_attn[mask]
            row["pearson_relax_vs_attn"] = pearson(ra, ea)
            row["spearman_relax_vs_attn"] = spearman(ra, ea)
            pooled["esm2_attn"].append(ea)
        per_protein.append(row)

        pooled["relax_act"].append(ra)
        pooled["final_conf"].append(fc)
        pooled["is_contact"].append(ct)
        pooled["gt_dist"].append(p["gt_dist"][mask])
        # seq separation
        idx = np.arange(L)
        seq_sep = np.abs(idx[:, None] - idx[None, :])[mask].astype(np.float32)
        pooled["seq_sep"].append(seq_sep)

    if not per_protein:
        log.error("No valid proteins.")
        return

    # Pooled across all val pairs (drop empty keys)
    pool = {k: np.concatenate(v) for k, v in pooled.items() if v}
    log.info("Pooled %d pairs across %d proteins", len(pool["relax_act"]), len(per_protein))

    pooled_corr = {
        "pearson_relax_vs_finalconf": pearson(pool["relax_act"], pool["final_conf"]),
        "spearman_relax_vs_finalconf": spearman(pool["relax_act"], pool["final_conf"]),
        "pearson_relax_vs_contact": pearson(pool["relax_act"], pool["is_contact"]),
        "spearman_relax_vs_contact": spearman(pool["relax_act"], pool["is_contact"]),
        "mean_relax_contact": float(pool["relax_act"][pool["is_contact"] > 0].mean()),
        "mean_relax_noncontact": float(pool["relax_act"][pool["is_contact"] == 0].mean()),
        "mean_relax_overall": float(pool["relax_act"].mean()),
    }
    if "esm2_attn" in pool:
        pooled_corr["pearson_relax_vs_attn"] = pearson(pool["relax_act"], pool["esm2_attn"])
        pooled_corr["spearman_relax_vs_attn"] = spearman(pool["relax_act"], pool["esm2_attn"])

    # ===== Distance-residualization: the real test =====
    log.info("\n=== Distance-residualized analysis ===")
    coef = np.polyfit(pool["gt_dist"], pool["relax_act"], deg=3)
    pred_r_from_d = np.polyval(coef, pool["gt_dist"])
    residual = pool["relax_act"] - pred_r_from_d
    log.info("  Fit cubic of relaxation~distance; residual std=%.2f (raw std=%.2f)",
              residual.std(), pool["relax_act"].std())
    pooled_corr["pearson_resid_vs_contact"] = pearson(residual, pool["is_contact"])
    pooled_corr["pearson_resid_vs_finalconf"] = pearson(residual, pool["final_conf"])
    pooled_corr["pearson_resid_vs_seqsep"] = pearson(residual, pool["seq_sep"])
    if "esm2_attn" in pool:
        pooled_corr["pearson_resid_vs_attn"] = pearson(residual, pool["esm2_attn"])
        pooled_corr["spearman_resid_vs_attn"] = spearman(residual, pool["esm2_attn"])
    log.info("  Distance-residualized correlations:")
    for k in ("pearson_resid_vs_attn", "spearman_resid_vs_attn",
               "pearson_resid_vs_contact", "pearson_resid_vs_finalconf",
               "pearson_resid_vs_seqsep"):
        if k in pooled_corr:
            log.info("    %-35s %+.4f", k, pooled_corr[k])
    log.info("=== Pooled correlations ===")
    for k, v in pooled_corr.items():
        log.info("  %-35s %+.4f" % (k, v) if isinstance(v, float) else "  %-35s %s" % (k, v))

    # Per-protein summary (mean ± std)
    log.info("\n=== Per-protein summary (mean ± std across %d proteins) ===", len(per_protein))
    for key in ["pearson_relax_vs_finalconf", "spearman_relax_vs_finalconf",
                 "pearson_relax_vs_attn", "pearson_relax_vs_contact"]:
        vals = [p[key] for p in per_protein if key in p and not np.isnan(p[key])]
        if vals:
            log.info("  %-35s %+.3f ± %.3f", key, np.mean(vals), np.std(vals))
    contact_ratios = [p["mean_relax_contact"] / max(p["mean_relax_noncontact"], 1e-9)
                       for p in per_protein]
    log.info("  mean_relax(contact)/mean_relax(non-contact): %.3f ± %.3f",
             np.mean(contact_ratios), np.std(contact_ratios))

    # Save JSON
    out = {"per_protein": per_protein, "pooled": pooled_corr}
    (RUNS_DIR / "importance_relaxation.json").write_text(json.dumps(out, indent=2))

    # ----- Plots -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    # Hexbin of relaxation_activity vs final_conf
    ax.hexbin(pool["final_conf"], pool["relax_act"], gridsize=40, cmap="Greys", bins="log", mincnt=1)
    ax.set_xlabel("final-block confidence (−entropy)", fontsize=10)
    ax.set_ylabel("total relaxation activity (sum |Δd| in non-metric phase)", fontsize=10)
    ax.set_title(f"Relaxation activity vs final confidence\nPearson r = {pooled_corr['pearson_relax_vs_finalconf']:+.3f}")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "esm2_attn" in pool:
        ax.hexbin(pool["esm2_attn"], pool["relax_act"], gridsize=40, cmap="Blues", bins="log", mincnt=1)
        ax.set_xlabel("mean ESM-2 attention at (i,j)", fontsize=10)
        ax.set_ylabel("relaxation activity", fontsize=10)
        ax.set_title(f"Relaxation vs ESM-2 attention\nPearson r = {pooled_corr['pearson_relax_vs_attn']:+.3f}")
    else:
        # Alternative: relaxation activity vs sequence separation
        ax.hexbin(pool["seq_sep"], pool["relax_act"], gridsize=40, cmap="Blues", bins="log", mincnt=1)
        sep_corr = pearson(pool["relax_act"], pool["seq_sep"])
        ax.set_xlabel("sequence separation |i−j|", fontsize=10)
        ax.set_ylabel("relaxation activity", fontsize=10)
        ax.set_title(f"Relaxation vs sequence separation\n(ESM-2 attention unavailable)  Pearson r = {sep_corr:+.3f}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    # Boxplot: relaxation activity for contact vs non-contact pairs
    data_box = [pool["relax_act"][pool["is_contact"] == 0],
                pool["relax_act"][pool["is_contact"] > 0]]
    bp = ax.boxplot(data_box, labels=["non-contact", "contact"], showfliers=False, widths=0.6,
                     patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgray")
    bp["boxes"][1].set_facecolor("steelblue")
    ax.set_ylabel("total relaxation activity (sum |Δd| in non-metric phase)", fontsize=10)
    ax.set_title(f"Relaxation activity: contact vs non-contact pairs\n"
                 f"mean: {pooled_corr['mean_relax_contact']:.2f} vs {pooled_corr['mean_relax_noncontact']:.2f}  "
                 f"(ratio: {pooled_corr['mean_relax_contact']/max(pooled_corr['mean_relax_noncontact'], 1e-9):.2f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # Per-protein bars showing each correlation
    names = [p["pdb_id"] for p in per_protein]
    pf = [p["pearson_relax_vs_finalconf"] for p in per_protein]
    pc = [p["pearson_relax_vs_contact"] for p in per_protein]
    has_attn = all("pearson_relax_vs_attn" in p for p in per_protein)
    x = np.arange(len(names))
    if has_attn:
        pa = [p["pearson_relax_vs_attn"] for p in per_protein]
        w = 0.27
        ax.bar(x - w, pf, w, label="vs final confidence", color="C0")
        ax.bar(x, pa, w, label="vs ESM-2 attention", color="C2")
        ax.bar(x + w, pc, w, label="vs is_contact", color="C3")
    else:
        w = 0.40
        ax.bar(x - w/2, pf, w, label="vs final confidence", color="C0")
        ax.bar(x + w/2, pc, w, label="vs is_contact", color="C3")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, fontsize=8)
    ax.set_ylabel("Pearson r (relaxation activity vs importance proxy)", fontsize=10)
    ax.set_title("Per-protein correlations")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Importance-weighted constraint relaxation hypothesis test\n"
                 f"(non-metric phase: block transitions {NONMETRIC_RANGE[0]} → {NONMETRIC_RANGE[1]})",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("runs/importance_relaxation.png", dpi=130, bbox_inches="tight")
    log.info("saved runs/importance_relaxation.png")


if __name__ == "__main__":
    main()
