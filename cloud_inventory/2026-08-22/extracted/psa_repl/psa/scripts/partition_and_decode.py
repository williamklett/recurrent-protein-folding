"""Properly test 'physics features → physical state' by:

  1. Partitioning SAE features per layer into three categories based on what they
     correlate with most strongly:
       physics  : best target is a structural one (DSSP class, K=2-derived geometry,
                  native per-residue physics like dihedrals/SASA)
       identity : best target is an amino-acid one-hot
       unaligned: nothing meets the quality threshold

  2. Training a SEPARATE decoder per category for each target:
       SS trace (per-residue, 8-class)
       contact map (per-pair, binary, via element-wise feature product)

  3. Also training: ALL features (upper bound) and RAW ESM-2 activations (baseline).

  Output:
     runs/feature_partition_layer<N>.json     which feature indices fall in each group
     runs/decoder_summary_partitioned.csv     one row per (layer, target, input_set)
"""
from __future__ import annotations

import csv
import json
import logging
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("partition_and_decode")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
ACT_DIR = ROOT / "data" / "esm2_650m_sae"
RUNS = ROOT / "runs"

LAYERS = [9, 24, 30]
SEED = 0

# Quality thresholds: a feature qualifies as "physical" (or "identity") only if its
# best target in that category meets these bars. Anything below is "unaligned."
PHYSICS_R2_THRESHOLD = 0.05        # any K=2 / native continuous R² above this counts
PHYSICS_AUC_THRESHOLD = 0.65       # any DSSP / contact AUC above this counts
IDENTITY_AUC_THRESHOLD = 0.85      # AA detector AUC threshold

PHYSICS_CONT_NAMES = {
    "burial", "mean_dist_to_5nn", "n_contacts",
    "mean_orient_cos_with_contacts", "orient_variance_with_neighbors",
    "phi_sin", "phi_cos", "psi_sin", "psi_cos", "omega_sin", "omega_cos", "sasa",
}
PHYSICS_BIN_NAMES = {
    "dssp_H", "dssp_B", "dssp_E", "dssp_G", "dssp_I", "dssp_T", "dssp_S", "dssp_coil",
    "is_contact",
}


# ============================================================
# Feature partitioning
# ============================================================
def partition_features(layer: int) -> dict:
    """Return {'physics': [feat_ids], 'identity': [feat_ids], 'unaligned': [feat_ids]}.

    A feature is 'physics' if any physics-vocab item exceeds its threshold.
    A feature is 'identity' if any aa_* AUC exceeds its threshold AND it didn't
    already qualify as physics (we prioritize physics when both qualify, since
    the question is whether physics-aligned features alone suffice).
    """
    d = np.load(RUNS / f"sae_feature_scores_layer{layer}.npz")
    r2 = d["r2_cont"]               # (n_feat, n_cont)
    auc = d["auc_sym"]              # (n_feat, n_bin)
    cont_names = list(d["cont_names"])
    bin_names = list(d["bin_names"])

    # Replace NaN AUCs (e.g. all-zero target columns) with 0 so they never qualify.
    auc = np.where(np.isnan(auc), 0.0, auc)

    # Per-feature best score in each physics-eligible subset
    phys_cont_idx = [i for i, n in enumerate(cont_names) if n in PHYSICS_CONT_NAMES]
    phys_bin_idx = [i for i, n in enumerate(bin_names) if n in PHYSICS_BIN_NAMES]
    aa_bin_idx = [i for i, n in enumerate(bin_names) if n.startswith("aa_")]

    best_phys_r2 = r2[:, phys_cont_idx].max(axis=1) if phys_cont_idx else np.zeros(r2.shape[0])
    best_phys_auc = auc[:, phys_bin_idx].max(axis=1) if phys_bin_idx else np.zeros(r2.shape[0])
    best_aa_auc = auc[:, aa_bin_idx].max(axis=1) if aa_bin_idx else np.zeros(r2.shape[0])

    is_physics = (best_phys_r2 >= PHYSICS_R2_THRESHOLD) | (best_phys_auc >= PHYSICS_AUC_THRESHOLD)
    is_identity = (best_aa_auc >= IDENTITY_AUC_THRESHOLD) & (~is_physics)

    out = {
        "physics": np.where(is_physics)[0].tolist(),
        "identity": np.where(is_identity)[0].tolist(),
        "unaligned": np.where(~(is_physics | is_identity))[0].tolist(),
    }
    out["thresholds"] = {
        "physics_r2": PHYSICS_R2_THRESHOLD,
        "physics_auc": PHYSICS_AUC_THRESHOLD,
        "identity_auc": IDENTITY_AUC_THRESHOLD,
    }
    return out


# ============================================================
# Data loading
# ============================================================
def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_residue_features(pdb_ids, layer, source: str, feat_mask: np.ndarray | None):
    """source ∈ {'sae', 'raw'}.  feat_mask selects columns of the SAE feature matrix
    (ignored for source='raw').  Returns (X, y_ss)."""
    Xs, ys = [], []
    for pdb_id in pdb_ids:
        sae = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        L = sae["L"]
        phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
        if phys["distances"].shape[0] != L:
            continue
        if source == "sae":
            X = sae["features"][layer].float().numpy()
            if feat_mask is not None:
                X = X[:, feat_mask]
        else:
            X = sae["activations"][layer].float().numpy()
        Xs.append(X)
        ys.append(phys["ss"].astype(int))
    return np.concatenate(Xs), np.concatenate(ys)


def load_pair_features(pdb_ids, layer, source: str, feat_mask: np.ndarray | None):
    rng = np.random.default_rng(SEED)
    Xs, ys = [], []
    for pdb_id in pdb_ids:
        sae = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        L = sae["L"]
        phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
        if phys["distances"].shape[0] != L:
            continue
        if source == "sae":
            F = sae["features"][layer].float().numpy()
            if feat_mask is not None:
                F = F[:, feat_mask]
        else:
            F = sae["activations"][layer].float().numpy()
        contacts = phys["contacts"]
        triu_i, triu_j = np.triu_indices(L, k=6)
        c_vals = contacts[triu_i, triu_j].astype(bool)
        pos = np.where(c_vals)[0]
        neg = np.where(~c_vals)[0]
        n_pos = min(len(pos), 300)
        n_neg = min(n_pos, len(neg))
        if n_pos == 0 or n_neg == 0:
            continue
        sel = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                rng.choice(neg, n_neg, replace=False)])
        ii, jj = triu_i[sel], triu_j[sel]
        Xs.append(F[ii] * F[jj])
        ys.append(np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32))
    return np.concatenate(Xs), np.concatenate(ys)


# ============================================================
# Fit / eval
# ============================================================
def fit_eval_ss(X_train, y_train, X_val, y_val) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    present = np.unique(y_train)
    mask_t = np.isin(y_train, present)
    mask_v = np.isin(y_val, present)
    clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    clf.fit(X_train[mask_t], y_train[mask_t])
    pred = clf.predict(X_val[mask_v])
    return {
        "accuracy": float(accuracy_score(y_val[mask_v], pred)),
        "macro_f1": float(f1_score(y_val[mask_v], pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val[mask_v], pred, average="weighted", zero_division=0)),
    }


def fit_eval_contact(X_train, y_train, X_val, y_val) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    return {
        "auc_roc": float(roc_auc_score(y_val, proba)),
        "auc_pr": float(average_precision_score(y_val, proba)),
        "acc": float(((proba > 0.5).astype(int) == y_val).mean()),
    }


def main():
    np.random.seed(SEED); random.seed(SEED)
    RUNS.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows]
    train_ids, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Split: %d train / %d val proteins", len(train_ids), len(val_ids))

    summary: list[dict] = []
    for layer in LAYERS:
        log.info("\n========= Layer %d =========", layer)
        part = partition_features(layer)
        n_phys = len(part["physics"])
        n_id = len(part["identity"])
        n_un = len(part["unaligned"])
        log.info("Partition: %4d physics  %4d identity  %4d unaligned  (total %d)",
                  n_phys, n_id, n_un, n_phys + n_id + n_un)
        (RUNS / f"feature_partition_layer{layer}.json").write_text(json.dumps(part))

        # Per-feature subsets as boolean masks (length 10240)
        phys_mask = np.zeros(10240, dtype=bool); phys_mask[part["physics"]] = True
        id_mask = np.zeros(10240, dtype=bool); id_mask[part["identity"]] = True
        all_mask = np.ones(10240, dtype=bool)

        configs = [
            ("physics_features",  "sae", phys_mask),
            ("identity_features", "sae", id_mask),
            ("all_features",      "sae", all_mask),
            ("raw_activations",   "raw", None),
        ]

        # SS decoder
        for label, source, mask in configs:
            n_in = int(mask.sum()) if source == "sae" else 1280
            if n_in == 0:
                log.warning("  SS | %s | skip (no features)", label)
                continue
            X_tr, y_tr = load_residue_features(train_ids, layer, source, mask)
            X_va, y_va = load_residue_features(val_ids, layer, source, mask)
            r = fit_eval_ss(X_tr, y_tr, X_va, y_va)
            log.info("  SS | %-18s | n_in=%5d | acc=%.3f  w_f1=%.3f  macro_f1=%.3f",
                      label, n_in, r["accuracy"], r["weighted_f1"], r["macro_f1"])
            summary.append({"layer": layer, "target": "ss", "input": label,
                             "n_input_features": n_in, "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                             **r})

        # Contact decoder
        for label, source, mask in configs:
            n_in = int(mask.sum()) if source == "sae" else 1280
            if n_in == 0:
                log.warning("  Contact | %s | skip", label)
                continue
            X_tr, y_tr = load_pair_features(train_ids, layer, source, mask)
            X_va, y_va = load_pair_features(val_ids, layer, source, mask)
            r = fit_eval_contact(X_tr, y_tr, X_va, y_va)
            log.info("  Contact | %-15s | n_in=%5d | AUC=%.3f  AP=%.3f  acc=%.3f",
                      label, n_in, r["auc_roc"], r["auc_pr"], r["acc"])
            summary.append({"layer": layer, "target": "contact", "input": label,
                             "n_input_features": n_in, "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                             **r})

    cols = ["layer", "target", "input", "n_input_features", "n_train", "n_val",
             "accuracy", "macro_f1", "weighted_f1", "auc_roc", "auc_pr", "acc"]
    out_path = RUNS / "decoder_summary_partitioned.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in summary:
            w.writerow({k: row.get(k, "") for k in cols})
    log.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
