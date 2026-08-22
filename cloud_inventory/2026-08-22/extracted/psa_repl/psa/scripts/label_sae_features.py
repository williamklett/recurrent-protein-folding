"""For every SAE feature at layers 9/24/30, score how well it correlates with
each item in three physical vocabularies. Produces:

  runs/sae_feature_labels.csv         one row per (layer, feature) with top label
  runs/sae_feature_scores_layer<N>.npz  full (n_features, n_targets) score matrices

Vocabularies:
  continuous K=2-derived (per-residue, 5 scalars)
  continuous native       (per-residue, 7 scalars: phi/psi/omega sin+cos, sasa)
  discrete                (per-residue: DSSP 8-class, AA 20-class, is_contact)

Continuous → continuous: Pearson r (we square at the end → R²)
Continuous (feature) → binary (target): AUC-ROC

We pool residues across all 53 proteins. Each per-protein activation file is
already (L, 10240) post-CLS/EOS stripping, aligned to the per-residue physics.
"""
from __future__ import annotations

import csv
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("label")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
ACT_DIR = ROOT / "data" / "esm2_650m_sae"
RUNS = ROOT / "runs"

LAYERS = [9, 24, 30]
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DSSP_LABELS = ["H", "B", "E", "G", "I", "T", "S", " ", "NA"]


def load_per_protein(pdb_id: str) -> dict:
    """Load one protein's physics + SAE features. Returns aligned per-residue."""
    sae = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
    phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
    L_sae = sae["L"]
    L_phys = phys["distances"].shape[0]
    if L_sae != L_phys:
        return None
    return {"sae": sae, "phys": phys, "L": L_sae}


def build_target_matrix(records: list[dict]) -> tuple[OrderedDict, np.ndarray]:
    """Stack per-residue physical targets across all proteins.

    Returns:
      schema : OrderedDict mapping target name → ("cont" or "bin")
      Y      : (n_residues_total, n_targets) float32 — for "bin" cols values are 0/1
    """
    schema: OrderedDict[str, str] = OrderedDict()
    # K=2-derived continuous
    for name in ["burial", "mean_dist_to_5nn", "n_contacts",
                  "mean_orient_cos_with_contacts", "orient_variance_with_neighbors"]:
        schema[name] = "cont"
    # Native continuous
    for name in ["phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin",
                  "omega_cos", "sasa"]:
        schema[name] = "cont"
    # Discrete one-hot
    for c in DSSP_LABELS:
        schema[f"dssp_{c.strip() or 'coil'}"] = "bin"
    for aa in AA_ALPHABET:
        schema[f"aa_{aa}"] = "bin"
    schema["is_contact"] = "bin"

    cols = list(schema.keys())
    n_total = sum(r["L"] for r in records)
    Y = np.zeros((n_total, len(cols)), dtype=np.float32)

    row = 0
    for r in records:
        phys = r["phys"]
        L = r["L"]
        sl = slice(row, row + L)

        # K=2-derived
        Y[sl, cols.index("burial")] = phys["burial"]
        Y[sl, cols.index("mean_dist_to_5nn")] = phys["mean_dist_to_5nn"]
        Y[sl, cols.index("n_contacts")] = phys["n_contacts"].astype(np.float32)
        Y[sl, cols.index("mean_orient_cos_with_contacts")] = phys["mean_orient_cos_with_contacts"]
        Y[sl, cols.index("orient_variance_with_neighbors")] = phys["orient_variance_with_neighbors"]

        # Native continuous
        Y[sl, cols.index("phi_sin")] = phys["phi"][:, 0]
        Y[sl, cols.index("phi_cos")] = phys["phi"][:, 1]
        Y[sl, cols.index("psi_sin")] = phys["psi"][:, 0]
        Y[sl, cols.index("psi_cos")] = phys["psi"][:, 1]
        Y[sl, cols.index("omega_sin")] = phys["omega"][:, 0]
        Y[sl, cols.index("omega_cos")] = phys["omega"][:, 1]
        Y[sl, cols.index("sasa")] = phys["sasa"]

        # DSSP 8-class (one-hot)
        ss = phys["ss"]
        for i, c in enumerate(DSSP_LABELS):
            Y[sl, cols.index(f"dssp_{c.strip() or 'coil'}")] = (ss == i).astype(np.float32)

        # AA identity (one-hot from sequence; np.savez stores str as 0-d array)
        seq = str(phys["sequence"])
        for i, ch in enumerate(seq[:L]):
            if ch in AA_ALPHABET:
                Y[row + i, cols.index(f"aa_{ch}")] = 1.0

        # is_contact = (n_contacts >= 1)
        Y[sl, cols.index("is_contact")] = (phys["n_contacts"] >= 1).astype(np.float32)

        row += L

    return schema, Y


def build_feature_matrix(records: list[dict], layer: int) -> np.ndarray:
    """Stack per-residue SAE features across all proteins for one layer.
    Returns (n_residues_total, 10240) fp32."""
    n_total = sum(r["L"] for r in records)
    X = np.zeros((n_total, 10240), dtype=np.float32)
    row = 0
    for r in records:
        z = r["sae"]["features"][layer].float().numpy()
        X[row : row + r["L"]] = z
        row += r["L"]
    return X


def pearson_columns(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between every column of X and every column of Y.
    Returns (n_x, n_y) matrix.

    X: (n, d_x), Y: (n, d_y). Both demeaned/normalized internally.
    """
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    x_std = np.sqrt((Xc * Xc).sum(0))
    y_std = np.sqrt((Yc * Yc).sum(0))
    # Avoid /0 for constant columns
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_std = np.where(y_std > 0, y_std, 1.0)
    return (Xc.T @ Yc) / (x_std[:, None] * y_std[None, :])


def auc_columns(X: np.ndarray, Y_bin: np.ndarray) -> np.ndarray:
    """AUC-ROC between every column of X (scores) and every column of Y_bin.
    Returns (n_x, n_y) matrix. Uses Mann-Whitney U: AUC = (R_pos - n_pos(n_pos+1)/2) / (n_pos n_neg)."""
    n = X.shape[0]
    # Rank each feature column once; same ranks for all targets
    # (Ranks here use 'average' for ties; we use scipy if available, else argsort)
    from scipy.stats import rankdata
    ranks = np.zeros_like(X, dtype=np.float32)
    for j in range(X.shape[1]):
        ranks[:, j] = rankdata(X[:, j])
    out = np.zeros((X.shape[1], Y_bin.shape[1]), dtype=np.float32)
    for t in range(Y_bin.shape[1]):
        y = Y_bin[:, t]
        n_pos = y.sum()
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            out[:, t] = np.nan
            continue
        # For each feature column, R_pos = sum of ranks where y==1
        # ranks has shape (n, n_x); broadcast y as (n, 1)
        R_pos = (ranks * y[:, None]).sum(0)
        out[:, t] = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    log.info("Loading per-protein data…")
    records: list[dict] = []
    for r in rows:
        rec = load_per_protein(r["pdb_id"])
        if rec is not None:
            records.append(rec)
        else:
            log.warning("Skipping %s (L mismatch)", r["pdb_id"])
    log.info("Loaded %d proteins (%d residues total)",
              len(records), sum(r["L"] for r in records))

    log.info("Building per-residue target matrix…")
    schema, Y = build_target_matrix(records)
    cont_idx = np.array([i for i, k in enumerate(schema) if schema[k] == "cont"])
    bin_idx = np.array([i for i, k in enumerate(schema) if schema[k] == "bin"])
    target_names = list(schema.keys())
    log.info("  Y: %s  (%d cont, %d bin)", Y.shape, len(cont_idx), len(bin_idx))

    all_rows: list[dict] = []
    for layer in LAYERS:
        log.info("Layer %d: building feature matrix…", layer)
        X = build_feature_matrix(records, layer)
        log.info("  X: %s, sparsity (frac nonzero) = %.3f", X.shape, (X > 0).mean())

        log.info("  Computing Pearson r vs continuous targets…")
        corr = pearson_columns(X, Y[:, cont_idx])         # (10240, n_cont)
        r2_cont = corr ** 2                                # R² per (feature, cont)

        log.info("  Computing AUC vs binary targets…")
        auc_bin = auc_columns(X, Y[:, bin_idx])           # (10240, n_bin)
        # Symmetric AUC: convert to "directional" max(auc, 1-auc) so we can compare
        # features that fire FOR a class vs AGAINST it.
        auc_sym = np.maximum(auc_bin, 1 - auc_bin)

        np.savez_compressed(
            RUNS / f"sae_feature_scores_layer{layer}.npz",
            r2_cont=r2_cont,
            auc_bin=auc_bin,
            auc_sym=auc_sym,
            cont_names=np.array([target_names[i] for i in cont_idx]),
            bin_names=np.array([target_names[i] for i in bin_idx]),
        )

        cont_names = [target_names[i] for i in cont_idx]
        bin_names = [target_names[i] for i in bin_idx]

        # For each feature, find best continuous + best binary target.
        best_cont_idx = r2_cont.argmax(axis=1)
        best_cont_val = r2_cont.max(axis=1)
        best_bin_idx = auc_sym.argmax(axis=1)
        best_bin_val = auc_sym.max(axis=1)

        # Per-feature: pick overall top with a heuristic — AUC and R² aren't directly
        # comparable but both range ~0.5–1.0 for "meaningful." Use rule:
        # if best_bin_val ≥ 0.8 prefer binary, else regression.
        n_active_per_feat = (X > 0).sum(0)
        n_active_proteins_per_feat = np.array([
            (X[sum(r["L"] for r in records[:i]):
              sum(r["L"] for r in records[:i + 1])] > 0).any(0)
            for i in range(len(records))
        ]).sum(0)
        for f in range(X.shape[1]):
            r2 = float(best_cont_val[f])
            auc = float(best_bin_val[f])
            top_cont = cont_names[best_cont_idx[f]]
            top_bin = bin_names[best_bin_idx[f]]
            # Pick "overall top" — prefer the higher informativeness.
            if auc >= 0.85 and (auc - 0.5) > r2:
                overall_kind, overall_label, overall_val = "bin", top_bin, auc
            else:
                overall_kind, overall_label, overall_val = "cont(R²)", top_cont, r2
            all_rows.append(
                {
                    "layer": layer,
                    "feature": f,
                    "top_cont": top_cont, "top_cont_R2": round(r2, 4),
                    "top_bin": top_bin, "top_bin_AUC": round(auc, 4),
                    "overall_kind": overall_kind, "overall_label": overall_label,
                    "overall_score": round(overall_val, 4),
                    "n_active_residues": int(n_active_per_feat[f]),
                    "n_active_proteins": int(n_active_proteins_per_feat[f]),
                }
            )
        log.info("  Done layer %d", layer)

    out_csv = RUNS / "sae_feature_labels.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    log.info("Wrote %s with %d rows", out_csv, len(all_rows))

    # Print headline: top 10 features per layer by overall score
    for layer in LAYERS:
        log.info("\n=== Top features at layer %d ===", layer)
        sub = [r for r in all_rows if r["layer"] == layer]
        sub.sort(key=lambda r: -r["overall_score"])
        log.info("%-6s %-8s %-25s %-8s  %-5s%-6s",
                  "feat", "kind", "label", "score", "act_r", "act_p")
        for r in sub[:10]:
            log.info("%-6d %-8s %-25s %-8.3f  %-5d%-6d",
                      r["feature"], r["overall_kind"], r["overall_label"],
                      r["overall_score"], r["n_active_residues"], r["n_active_proteins"])


if __name__ == "__main__":
    main()
