"""Train linear decoders from SAE features to physical state.

Two targets:
  1. Per-residue 8-class DSSP secondary structure
  2. Per-pair contact probability (using element-wise product of per-residue features)

Both use logistic regression. We compare:
  * decoder on ALL 10240 SAE features (the upper bound of linear accessibility)
  * decoder on the top-K features by individual relevance (a "compressed" version)
  * decoder on the RAW ESM-2 activations as a baseline (does the SAE add information
    or just rearrange it?)

Per-protein 80/20 split (NOT per-residue) so the eval is on unseen proteins.

Outputs:
  runs/decoder_summary.csv          one row per (layer, target, model)
  runs/decoder_models/<layer>_<target>_<model>.pkl
"""
from __future__ import annotations

import csv
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("decode")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
ACT_DIR = ROOT / "data" / "esm2_650m_sae"
RUNS = ROOT / "runs"
MODEL_DIR = RUNS / "decoder_models"

LAYERS = [9, 24, 30]
SEED = 0
DSSP_LABELS = ["H", "B", "E", "G", "I", "T", "S", " ", "NA"]


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_residue_data(pdb_ids, layer, use_sae: bool):
    """Stack per-residue features + DSSP labels across the given proteins.
    Returns (X, y_ss, group_id, protein_lengths)."""
    Xs, ys, gs = [], [], []
    lengths = []
    for pidx, pdb_id in enumerate(pdb_ids):
        sae = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        L = sae["L"]
        phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
        if phys["distances"].shape[0] != L:
            continue
        if use_sae:
            X = sae["features"][layer].float().numpy()    # (L, 10240)
        else:
            X = sae["activations"][layer].float().numpy()  # (L, 1280)
        ss = phys["ss"].astype(int)
        Xs.append(X)
        ys.append(ss)
        gs.append(np.full(L, pidx))
        lengths.append(L)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(gs, axis=0)
    return X, y, g, lengths


def load_pair_data(pdb_ids, layer, max_pairs_per_protein: int = 600):
    """Build pair features for contact prediction.

    For each protein, sample up to max_pairs_per_protein pairs (i, j) with i < j and |i-j| > 5
    (skip short-range trivial pairs), balanced 50/50 positive/negative.
    Pair feature = f_i * f_j (element-wise product of per-residue SAE features).

    Returns X_pairs, y_contact, protein_group.
    """
    rng = np.random.default_rng(SEED)
    Xs, ys, gs = [], [], []
    for pidx, pdb_id in enumerate(pdb_ids):
        sae = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        L = sae["L"]
        phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
        if phys["distances"].shape[0] != L:
            continue
        F = sae["features"][layer].float().numpy()    # (L, 10240) fp32
        contacts = phys["contacts"]
        seq_sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])

        triu_i, triu_j = np.triu_indices(L, k=6)  # i < j, |i-j| > 5
        c_vals = contacts[triu_i, triu_j].astype(bool)
        pos_idx = np.where(c_vals)[0]
        neg_idx = np.where(~c_vals)[0]
        n_pos = min(len(pos_idx), max_pairs_per_protein // 2)
        n_neg = n_pos
        if n_pos == 0 or len(neg_idx) == 0:
            continue
        pos_sel = rng.choice(pos_idx, n_pos, replace=False)
        neg_sel = rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
        sel = np.concatenate([pos_sel, neg_sel])
        ii, jj = triu_i[sel], triu_j[sel]
        pair_feat = F[ii] * F[jj]  # (n_sel, 10240)
        y_sel = np.concatenate([np.ones(n_pos), np.zeros(len(neg_sel))]).astype(np.float32)
        Xs.append(pair_feat)
        ys.append(y_sel)
        gs.append(np.full(len(sel), pidx))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(gs, axis=0)
    return X, y, g


# ============================================================
# Logistic regression / multinomial classifier — use sklearn
# ============================================================
def fit_eval_ss(X_train, y_train, X_val, y_val) -> dict:
    """Multinomial logistic regression on 8-class DSSP."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    # Drop classes that aren't present in train (DSSP 'NA' / 'I' may be empty)
    present = np.unique(y_train)
    mask_t = np.isin(y_train, present)
    mask_v = np.isin(y_val, present)
    clf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1, C=1.0)
    clf.fit(X_train[mask_t], y_train[mask_t])
    pred = clf.predict(X_val[mask_v])
    return {
        "accuracy": float(accuracy_score(y_val[mask_v], pred)),
        "macro_f1": float(f1_score(y_val[mask_v], pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val[mask_v], pred, average="weighted", zero_division=0)),
        "n_classes": int(len(present)),
        "n_features": X_train.shape[1],
        "n_train": int(mask_t.sum()),
        "n_val": int(mask_v.sum()),
        "_model": clf,
    }


def fit_eval_contact(X_train, y_train, X_val, y_val) -> dict:
    """Binary logistic regression on contact label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    clf = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1, C=1.0)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    return {
        "auc_roc": float(roc_auc_score(y_val, proba)),
        "auc_pr": float(average_precision_score(y_val, proba)),
        "accuracy_at_0.5": float((((proba > 0.5).astype(int) == y_val).mean())),
        "base_rate": float(y_val.mean()),
        "n_features": X_train.shape[1],
        "n_train": len(y_train),
        "n_val": len(y_val),
        "_model": clf,
    }


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows]
    train_ids, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Split: %d train / %d val proteins", len(train_ids), len(val_ids))
    log.info("Val IDs: %s", val_ids)

    summary_rows: list[dict] = []
    for layer in LAYERS:
        log.info("\n========= Layer %d =========", layer)

        # --- Per-residue: SS decoder ---
        for use_sae, model_name in [(True, "sae_features"), (False, "raw_activations")]:
            X_tr, y_tr, _, _ = load_residue_data(train_ids, layer, use_sae)
            X_va, y_va, _, _ = load_residue_data(val_ids, layer, use_sae)
            log.info("  SS decoder | %s | X_tr=%s, X_va=%s",
                      model_name, X_tr.shape, X_va.shape)
            r = fit_eval_ss(X_tr, y_tr, X_va, y_va)
            model_path = MODEL_DIR / f"layer{layer}_ss_{model_name}.pkl"
            with model_path.open("wb") as f:
                pickle.dump(r["_model"], f)
            summary_rows.append({
                "layer": layer, "target": "ss", "input": model_name,
                "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"], "auc_roc": "", "auc_pr": "",
                "n_features": r["n_features"], "n_train": r["n_train"], "n_val": r["n_val"],
            })
            log.info("    accuracy=%.3f  weighted_f1=%.3f  macro_f1=%.3f",
                      r["accuracy"], r["weighted_f1"], r["macro_f1"])

        # --- Per-pair: contact decoder ---
        for use_sae, model_name in [(True, "sae_features"), (False, "raw_activations")]:
            # For raw activations, build pair features the same way (element-wise product)
            X_tr, y_tr, _ = load_pair_data_dispatch(train_ids, layer, use_sae)
            X_va, y_va, _ = load_pair_data_dispatch(val_ids, layer, use_sae)
            log.info("  Contact decoder | %s | X_tr=%s, X_va=%s",
                      model_name, X_tr.shape, X_va.shape)
            r = fit_eval_contact(X_tr, y_tr, X_va, y_va)
            model_path = MODEL_DIR / f"layer{layer}_contact_{model_name}.pkl"
            with model_path.open("wb") as f:
                pickle.dump(r["_model"], f)
            summary_rows.append({
                "layer": layer, "target": "contact", "input": model_name,
                "accuracy": "", "macro_f1": "", "weighted_f1": "",
                "auc_roc": r["auc_roc"], "auc_pr": r["auc_pr"],
                "n_features": r["n_features"], "n_train": r["n_train"], "n_val": r["n_val"],
            })
            log.info("    AUC-ROC=%.3f  AUC-PR=%.3f  acc@0.5=%.3f  (base=%.3f)",
                      r["auc_roc"], r["auc_pr"], r["accuracy_at_0.5"], r["base_rate"])

    csv_path = RUNS / "decoder_summary.csv"
    cols = ["layer", "target", "input", "accuracy", "macro_f1", "weighted_f1",
            "auc_roc", "auc_pr", "n_features", "n_train", "n_val"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    log.info("\nWrote %s", csv_path)


def load_pair_data_dispatch(pdb_ids, layer, use_sae: bool):
    """Build pair features using either SAE features or raw ESM-2 activations."""
    rng = np.random.default_rng(SEED)
    Xs, ys, gs = [], [], []
    for pidx, pdb_id in enumerate(pdb_ids):
        sae = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        L = sae["L"]
        phys = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
        if phys["distances"].shape[0] != L:
            continue
        F = (sae["features"][layer] if use_sae else sae["activations"][layer]).float().numpy()
        contacts = phys["contacts"]
        triu_i, triu_j = np.triu_indices(L, k=6)
        c_vals = contacts[triu_i, triu_j].astype(bool)
        pos_idx = np.where(c_vals)[0]
        neg_idx = np.where(~c_vals)[0]
        n_pos = min(len(pos_idx), 300)
        n_neg = min(n_pos, len(neg_idx))
        if n_pos == 0 or n_neg == 0:
            continue
        pos_sel = rng.choice(pos_idx, n_pos, replace=False)
        neg_sel = rng.choice(neg_idx, n_neg, replace=False)
        sel = np.concatenate([pos_sel, neg_sel])
        ii, jj = triu_i[sel], triu_j[sel]
        pair_feat = F[ii] * F[jj]
        Xs.append(pair_feat)
        ys.append(np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32))
        gs.append(np.full(len(sel), pidx))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


if __name__ == "__main__":
    main()
