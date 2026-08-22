"""Characterize the per-layer GEOMETRIC EDITS the trunk makes, via the
distogram-head decoder applied to every block.

Six measures, computed across val proteins:

  1. magnitude profile   per block: mean |edit|, max |edit|, % edits > 1 Å
  2. directionality      per block: mean SIGNED edit (overall, contacts-only,
                                     non-contacts-only)
  3. localization        per block: Gini coefficient of |edit| (1 = all change
                                     in one pair; 0 = uniform)
  4. edit modes (PCA)    per block: variance fraction explained by top-1 PC
                                     of per-pair edits across proteins
  5. convergence-to-truth per block:
        — fraction of pairs where sign(edit) == sign(true_dist - pred_dist_b)
        — Pearson r between edit and (true_dist - pred_dist_b)
  6. triangle-inequality per block:
        — fraction of random triangles (i,j,k) where d(i,k) > d(i,j)+d(j,k)
          (violations); does it shrink with depth?

Output:
  runs/geometric_edits_summary.json
  runs/geometric_edits.png
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("geom_edits")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
DISTO_DIR = ROOT / "data" / "distograms"
RUNS_DIR = ROOT / "runs"

SEED = 0
N_BLOCKS = 48


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_protein(pdb_id: str) -> dict:
    d = torch.load(DISTO_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
    phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
    pred = d["pred_distance"].float().numpy()      # (48, L, L)
    L = pred.shape[1]
    gt = phys["distances"].astype(np.float32)
    if gt.shape[0] != L:
        return None
    contacts = phys["contacts"].astype(bool)
    eye = np.eye(L, dtype=bool)
    return {"pdb_id": pdb_id, "L": L, "pred": pred, "gt": gt,
             "contacts": contacts, "eye": eye}


def gini(x: np.ndarray) -> float:
    """Gini coefficient of nonneg values. 0=uniform, 1=all mass in one element."""
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def main() -> None:
    np.random.seed(SEED); random.seed(SEED)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows
                if (DISTO_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    proteins = [p for p in (load_protein(pid) for pid in val_ids) if p is not None]
    log.info("Loaded %d proteins", len(proteins))

    # Per-transition results
    transitions = list(range(N_BLOCKS - 1))
    n_t = len(transitions)
    out = {
        "n_proteins": len(proteins),
        "blocks": list(range(N_BLOCKS)),
        "transitions": transitions,
        "magnitude": {"mean_abs": [], "max_abs": [], "frac_gt_1A": []},
        "directionality": {"mean_signed_all": [],
                            "mean_signed_contacts": [],
                            "mean_signed_noncontacts": []},
        "localization_gini": [],
        "edit_modes_top1_var": [],
        "convergence": {"frac_correct_sign": [], "corr_edit_vs_gap": []},
        "triangle_violation_fraction": [None] * N_BLOCKS,
    }

    # Pre-compute per-protein edit tensors once (memory ok at fp32)
    edits = []   # list of (n_transitions, L, L) per protein
    for p in proteins:
        e = p["pred"][1:] - p["pred"][:-1]    # (47, L, L)
        edits.append(e)

    # Per-transition aggregates
    for ti, t in enumerate(transitions):
        # Pool over proteins
        abs_vals = []
        signed_all = []
        signed_c = []
        signed_nc = []
        gini_per_prot = []
        # For PCA of edit modes — we use one flattened vector per protein
        pca_vecs = []
        # Convergence-to-truth
        frac_correct_per_prot = []
        corr_per_prot = []

        for pi, p in enumerate(proteins):
            e = edits[pi][t]                    # (L, L)
            L = p["L"]
            mask = ~p["eye"]
            ev = e[mask]
            abs_vals.append(np.abs(ev))
            signed_all.append(ev)
            cmask = p["contacts"] & mask
            ncmask = (~p["contacts"]) & mask
            signed_c.append(e[cmask])
            signed_nc.append(e[ncmask])
            gini_per_prot.append(gini(ev))
            # PCA mode: take fixed-size feature — variance of edits as a histogram
            # Histogram from -5 to +5 Å in 50 bins, normalize
            hist, _ = np.histogram(ev, bins=50, range=(-5, 5))
            pca_vecs.append(hist / max(1, hist.sum()))

            # Convergence to truth: gap = true - pred_BEFORE this transition
            gap = (p["gt"] - p["pred"][t])[mask]
            sign_correct = (np.sign(ev) == np.sign(gap)).astype(np.float32)
            frac_correct_per_prot.append(float(sign_correct.mean()))
            # Pearson r
            if ev.std() > 0 and gap.std() > 0:
                corr = float(np.corrcoef(ev, gap)[0, 1])
            else:
                corr = float("nan")
            corr_per_prot.append(corr)

        abs_concat = np.concatenate(abs_vals)
        out["magnitude"]["mean_abs"].append(float(abs_concat.mean()))
        out["magnitude"]["max_abs"].append(float(abs_concat.max()))
        out["magnitude"]["frac_gt_1A"].append(float((abs_concat > 1.0).mean()))

        out["directionality"]["mean_signed_all"].append(float(np.concatenate(signed_all).mean()))
        out["directionality"]["mean_signed_contacts"].append(float(np.concatenate(signed_c).mean()))
        out["directionality"]["mean_signed_noncontacts"].append(float(np.concatenate(signed_nc).mean()))

        out["localization_gini"].append(float(np.mean(gini_per_prot)))

        # PCA on edit-distribution histograms across proteins
        Pmat = np.stack(pca_vecs)
        Pc = Pmat - Pmat.mean(0, keepdims=True)
        # Use number of proteins (rows) as samples; SVD on (n_prot, n_bins)
        try:
            u, s, vt = np.linalg.svd(Pc, full_matrices=False)
            var_top1 = float((s[0] ** 2) / (s ** 2).sum()) if (s ** 2).sum() > 0 else float("nan")
        except Exception:
            var_top1 = float("nan")
        out["edit_modes_top1_var"].append(var_top1)

        out["convergence"]["frac_correct_sign"].append(float(np.mean(frac_correct_per_prot)))
        out["convergence"]["corr_edit_vs_gap"].append(float(np.nanmean(corr_per_prot)))

        if ti % 8 == 0:
            log.info("transition %2d→%2d  |  |Δd|=%.3f  signed=%+.3f  gini=%.3f  conv_sign=%.3f  conv_corr=%+.3f",
                      t, t + 1, out["magnitude"]["mean_abs"][-1],
                      out["directionality"]["mean_signed_all"][-1],
                      out["localization_gini"][-1],
                      out["convergence"]["frac_correct_sign"][-1],
                      out["convergence"]["corr_edit_vs_gap"][-1])

    # Triangle inequality fraction violation per BLOCK (not transition)
    log.info("Triangle inequality check…")
    n_triangles = 20_000
    for b in range(N_BLOCKS):
        viols = []
        for p in proteins:
            L = p["L"]
            if L < 4:
                continue
            d = p["pred"][b]
            # Sample n_triangles ordered triples (i, j, k), i < j < k
            idx = np.random.choice(L, size=(n_triangles, 3), replace=True)
            idx.sort(axis=1)
            idx_uniq = idx[(idx[:, 0] < idx[:, 1]) & (idx[:, 1] < idx[:, 2])]
            if len(idx_uniq) == 0:
                continue
            ii, jj, kk = idx_uniq[:, 0], idx_uniq[:, 1], idx_uniq[:, 2]
            d_ij = d[ii, jj]
            d_jk = d[jj, kk]
            d_ik = d[ii, kk]
            # Violations of triangle inequality: d_ik > d_ij + d_jk
            v1 = d_ik > d_ij + d_jk
            v2 = d_ij > d_ik + d_jk
            v3 = d_jk > d_ij + d_ik
            viols.append(float((v1 | v2 | v3).mean()))
        out["triangle_violation_fraction"][b] = float(np.mean(viols)) if viols else float("nan")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "geometric_edits_summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote runs/geometric_edits_summary.json")

    # ===== Plots =====
    t_arr = np.array(out["transitions"]) + 0.5   # midpoint between blocks for plotting
    b_arr = np.array(out["blocks"])
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.plot(t_arr, out["magnitude"]["mean_abs"], '-o', color='C3', markersize=3, label='mean |Δd|')
    ax.plot(t_arr, out["magnitude"]["frac_gt_1A"], '-s', color='C1', markersize=3, alpha=0.7, label='% pairs with |Δd|>1Å')
    ax.set_xlabel('layer transition b→b+1'); ax.set_ylabel('value')
    ax.set_title('1. Magnitude profile — how big are the per-layer distance edits?')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_arr, out["directionality"]["mean_signed_all"], '-o', color='black', markersize=3, label='all pairs')
    ax.plot(t_arr, out["directionality"]["mean_signed_contacts"], '-s', color='C0', markersize=3, label='contacts (true)')
    ax.plot(t_arr, out["directionality"]["mean_signed_noncontacts"], '-^', color='C3', markersize=3, label='non-contacts (true)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('transition'); ax.set_ylabel('mean signed Δd (Å)')
    ax.set_title('2. Directionality — are layers expanding or contracting?')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_arr, out["localization_gini"], '-o', color='C2', markersize=3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('transition'); ax.set_ylabel('Gini of |Δd|')
    ax.set_title('3. Localization — broadcast (Gini 0) vs targeted (Gini 1) edits')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_arr, out["edit_modes_top1_var"], '-o', color='C4', markersize=3)
    ax.set_xlabel('transition'); ax.set_ylabel('top-1 PC variance fraction')
    ax.set_title('4. Edit modes — does this layer apply ONE operation (top1 high) or many?')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t_arr, out["convergence"]["frac_correct_sign"], '-o', color='C0', markersize=3, label='fraction correct sign')
    ax.axhline(0.5, color='gray', lw=0.5, label='chance')
    ax.set_xlabel('transition'); ax.set_ylabel('fraction')
    ax.set_title('5a. Convergence to truth — does this layer push the right direction?')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    ax = axes[2, 1]
    ax.plot(t_arr, out["convergence"]["corr_edit_vs_gap"], '-o', color='C1', markersize=3, label='Pearson r(edit, true−pred_before)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('transition'); ax.set_ylabel('Pearson r')
    ax.set_title('5b. Correlation between edit and gap-to-truth — strength of refinement')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('runs/geometric_edits.png', dpi=130, bbox_inches='tight')
    log.info('saved runs/geometric_edits.png')

    # Separate triangle-inequality plot per BLOCK
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    tri = out["triangle_violation_fraction"]
    ax.plot(b_arr, tri, '-o', color='C5', markersize=3)
    ax.set_xlabel('block'); ax.set_ylabel('fraction of triangles violating triangle inequality')
    ax.set_title('6. Triangle-inequality satisfaction across depth\n(low = consistent distance map; how much does the trunk enforce metric structure?)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/triangle_inequality.png', dpi=130, bbox_inches='tight')
    log.info('saved runs/triangle_inequality.png')


if __name__ == "__main__":
    main()
