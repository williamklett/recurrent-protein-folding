"""K=2 PSA sweep across every ESMFold trunk block (pure-geometry bottleneck).

Compares directly to the K=5 result in runs/sweep_summary.json. Same protein
split, same architecture, same hyperparameters — only the bottleneck schema
changes (distance + orient_cos vs. K=5's distance + contact + h-bond +
log_seq_sep + orient_cos).
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import torch

from psa.models import PSA
from psa.targets_k2 import (
    FeatureStatsK2,
    K_PAIR_K2,
    PAIR_FEATURE_NAMES_K2,
    build_pair_targets_k2,
)
from psa.train import TrainConfig, evaluate, train_psa, r2_score

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")
log = logging.getLogger("sweep_k2")
log.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

NUM_BLOCKS = 48
LAMBDA_PHYS = 100.0
EPOCHS = 25
SEED = 0
D_PAIR = 128
D_HIDDEN = 128


def split_by_protein(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_block_data(block_idx: int, pdb_ids: list[str]):
    """Concatenate pair-rep activations and K=2 physics targets across proteins."""
    xs, ys = [], []
    for pdb_id in pdb_ids:
        act = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt",
                          map_location="cpu", weights_only=False)
        z = act["z"][block_idx, 0].float()           # (L, L, 128)
        L = z.shape[0]
        y_flat, keep = build_pair_targets_k2(PHYS_DIR / f"{pdb_id.lower()}.npz")
        L_phys = int(y_flat.shape[0] ** 0.5)
        if L_phys != L:
            log.warning("%s  L mismatch (esm=%d phys=%d); skipping", pdb_id, L, L_phys)
            continue
        x_flat = z.reshape(L * L, D_PAIR)
        xs.append(x_flat[keep])
        ys.append(y_flat[keep])
    return torch.cat(xs), torch.cat(ys)


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    train_ids, val_ids = split_by_protein(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("K=2 sweep: %d blocks, %d train / %d val proteins, lambda=%.1f",
             NUM_BLOCKS, len(train_ids), len(val_ids), LAMBDA_PHYS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict] = []
    t_start = time.time()

    for b in range(NUM_BLOCKS):
        t0 = time.time()
        X_train, y_train_raw = load_block_data(b, train_ids)
        X_val, y_val_raw = load_block_data(b, val_ids)
        stats = FeatureStatsK2.from_targets([y_train_raw])
        y_train_norm = stats.normalize(y_train_raw)
        y_val_norm = stats.normalize(y_val_raw)

        model = PSA(d_pair=D_PAIR, d_hidden=D_HIDDEN, K=K_PAIR_K2)
        cfg = TrainConfig(epochs=EPOCHS, batch_size=8192, lr=1e-3,
                           lambda_phys=LAMBDA_PHYS)
        history = train_psa(model, X_train, y_train_norm, X_val, y_val_norm,
                             stats, cfg=cfg, device=device,
                             feature_names=PAIR_FEATURE_NAMES_K2)
        final = history[-1].copy()
        final["block"] = b
        final["sec"] = time.time() - t0
        # Per-feature R² on val (denormalized)
        model.eval().to(device)
        with torch.no_grad():
            X_val_d = X_val.to(device)
            z_pred, h_hat = model(X_val_d)
        z_pred_raw = z_pred.cpu() * stats.std + stats.mean
        for k, name in enumerate(PAIR_FEATURE_NAMES_K2):
            final[f"r2_{name}_raw"] = r2_score(y_val_raw[:, k : k + 1],
                                                 z_pred_raw[:, k : k + 1])
        results.append(final)

        # Save per-block model
        block_dir = RUNS_DIR / f"k2_block_{b:02d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), block_dir / "model.pt")
        torch.save({"mean": stats.mean, "std": stats.std,
                     "is_binary": stats.is_binary}, block_dir / "stats.pt")

        log.info(
            "block %2d in %.1fs (total %.0fs) | recon=%.3f  d_R²=%.3f  orient_R²=%.3f",
            b, final["sec"], time.time() - t_start, final["r2_recon"],
            final["r2_distance_raw"], final["r2_orient_cos_raw"],
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNS_DIR / "k2_sweep_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    cols = list(results[0].keys())
    with (RUNS_DIR / "k2_sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    log.info("Wrote k2_sweep_summary.{json,csv}.  Total %.0fs.", time.time() - t_start)


if __name__ == "__main__":
    main()
