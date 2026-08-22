"""Properly evaluate the saved PSAs with threshold-free + distribution metrics.

The previous eval reported binary accuracy at threshold 0.5, which is uninformative
for highly imbalanced features (contact 9% positive, h-bond 1.2% positive — a
constant "no" predictor already scores 0.91 / 0.99).

This script computes for each block (ESMFold trunk) and each layer (ESM-2):
  Binary features:
    * AUC-ROC
    * AUC-PR (a.k.a. average precision) — the right metric for rare events
    * F1 at best threshold
    * Precision / recall at best F1 threshold
  Continuous features:
    * R² (unchanged)
    * Pearson r
    * predicted-vs-true sample for the val set (a few hundred random points)

Outputs:
  runs/proper_metrics.json
  runs/proper_metrics.csv
  runs/figs/binary_diagnostics_block47.png       — ROC + PR for h-bond and contact
  runs/figs/distance_scatter_block47.png         — pred vs true distance scatter
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from psa.models import PSA
from psa.targets import FeatureStats, K_PAIR, PAIR_FEATURE_SCHEMA, build_pair_targets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("evaluate_proper")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"
FIG_DIR = RUNS_DIR / "figs"

SEED = 0


# ---------- metrics ----------
def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC. y_true ∈ {0,1}, scores higher = more positive."""
    if y_true.min() == y_true.max():
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_tp = np.cumsum(y_sorted)
    fp = np.arange(1, len(y_sorted) + 1) - cum_tp
    # ROC AUC via Mann-Whitney
    ranks = np.empty_like(scores, dtype=float)
    ranks[order] = np.arange(len(scores), 0, -1)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under PR curve). Right metric for rare positives."""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    precision = cum_tp / np.arange(1, len(y_sorted) + 1)
    # AP = sum over positives of precision at that recall
    return float((precision * y_sorted).sum() / n_pos)


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Sweep thresholds, return (best F1, precision, recall, threshold)."""
    order = np.argsort(-scores)
    y_s = y_true[order]
    s_s = scores[order]
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    tp = np.cumsum(y_s)
    fp = np.arange(1, len(y_s) + 1) - tp
    fn = n_pos - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / n_pos
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1))
    return float(f1[best]), float(precision[best]), float(recall[best]), float(s_s[best])


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / (den + 1e-12))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ---------- data loading ----------
def split_by_protein(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_val_features(source: str, block_or_layer: int, val_ids: list[str]):
    """Concatenate val activations + targets. source ∈ {'trunk', 'esm2'}."""
    xs, ys = [], []
    for pdb_id in val_ids:
        if source == "trunk":
            cache = torch.load(ROOT / "data" / "activations" / f"{pdb_id.lower()}.pt",
                                map_location="cpu", weights_only=False)
            x = cache["z"][block_or_layer, 0].float()  # (L, L, 128)
        else:
            cache = torch.load(ROOT / "data" / "esm2_attention" / f"{pdb_id.lower()}.pt",
                                map_location="cpu", weights_only=False)
            x = cache["attn"][block_or_layer, 0].float().permute(1, 2, 0)  # (L, L, 80)
        L = x.shape[0]
        y_flat, keep = build_pair_targets(PHYS_DIR / f"{pdb_id.lower()}.npz")
        L_phys = int(y_flat.shape[0] ** 0.5)
        if L_phys != L:
            continue
        d_pair = x.shape[-1]
        xs.append(x.reshape(L * L, d_pair)[keep])
        ys.append(y_flat[keep])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def load_psa(run_dir: Path, d_pair: int) -> tuple[PSA, FeatureStats]:
    model = PSA(d_pair=d_pair, d_hidden=128, K=K_PAIR)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    s = torch.load(run_dir / "stats.pt", map_location="cpu", weights_only=True)
    return model, FeatureStats(mean=s["mean"], std=s["std"], is_binary=s["is_binary"])


def evaluate_one(source: str, idx: int, val_ids: list[str], d_pair: int) -> dict:
    if source == "trunk":
        run_dir = RUNS_DIR / f"block_{idx:02d}"
    else:
        run_dir = RUNS_DIR / f"esm2_layer_{idx:02d}"
    if not run_dir.exists():
        return None

    psa, stats = load_psa(run_dir, d_pair)
    X, y_raw = load_val_features(source, idx, val_ids)

    with torch.no_grad():
        z_pred, h_hat = psa(X)
    z_pred = z_pred.numpy()
    y = y_raw.numpy()
    h = X.numpy()
    h_hat = h_hat.numpy()

    out = {"source": source, "index": idx, "n_val_pairs": int(len(y))}
    out["r2_recon"] = r2(h.reshape(-1), h_hat.reshape(-1))

    for k, (name, kind) in enumerate(PAIR_FEATURE_SCHEMA):
        if kind == "bin":
            scores = z_pred[:, k]  # logits
            y_k = y[:, k].astype(int)
            out[f"{name}_auc_roc"] = auc_roc(y_k, scores)
            out[f"{name}_auc_pr"] = auc_pr(y_k, scores)
            f1, prec, rec, thr = best_f1(y_k, scores)
            out[f"{name}_best_f1"] = f1
            out[f"{name}_best_f1_precision"] = prec
            out[f"{name}_best_f1_recall"] = rec
            out[f"{name}_base_rate"] = float(y_k.mean())
        else:
            # Predictions are in normalized space. Denormalize for interpretability.
            pred_raw = z_pred[:, k] * float(stats.std[k]) + float(stats.mean[k])
            y_k = y[:, k]
            out[f"{name}_r2"] = r2(y_k, pred_raw)
            out[f"{name}_pearson_r"] = pearson_r(y_k, pred_raw)
    return out


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_by_protein(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val set: %s", val_ids)

    # ----- Trunk (48 blocks, d=128) and ESM-2 (36 layers, d=80) -----
    results = []
    for b in range(48):
        r = evaluate_one("trunk", b, val_ids, d_pair=128)
        if r:
            results.append(r)
            log.info("trunk block %2d | recon=%.3f c_auc=%.3f h_auc=%.3f c_pr=%.3f h_pr=%.3f d_r2=%.3f",
                      b, r["r2_recon"], r["contact_auc_roc"], r["hbond_auc_roc"],
                      r["contact_auc_pr"], r["hbond_auc_pr"], r["distance_r2"])

    for ℓ in range(36):
        r = evaluate_one("esm2", ℓ, val_ids, d_pair=80)
        if r:
            results.append(r)
            log.info("esm2 layer %2d | recon=%.3f c_auc=%.3f h_auc=%.3f c_pr=%.3f h_pr=%.3f d_r2=%.3f",
                      ℓ, r["r2_recon"], r["contact_auc_roc"], r["hbond_auc_roc"],
                      r["contact_auc_pr"], r["hbond_auc_pr"], r["distance_r2"])

    with (RUNS_DIR / "proper_metrics.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Write CSV with stable column order
    fieldnames = sorted({k for r in results for k in r.keys()})
    with (RUNS_DIR / "proper_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    log.info("wrote runs/proper_metrics.{json,csv}")


if __name__ == "__main__":
    main()
