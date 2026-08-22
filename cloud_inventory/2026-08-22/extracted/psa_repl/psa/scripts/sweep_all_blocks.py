"""Sweep PSA training across every trunk block, producing the R²-vs-depth curve.

Loads activations + physics once, then trains a fresh PSA per block. Writes:
    runs/sweep_summary.json      one row per block, all metrics
    runs/sweep_summary.csv       same, csv form
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
from psa.targets import K_PAIR, FeatureStats
from psa.train import TrainConfig, evaluate, train_psa

# Reuse the loader from train_psa_real
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_psa_real import load_block_data, split_by_protein  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")
log = logging.getLogger("sweep")
log.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
RUNS_DIR = ROOT / "runs"

NUM_BLOCKS = 48
LAMBDA_PHYS = 100.0
EPOCHS = 25
SEED = 0


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    train_ids, val_ids = split_by_protein(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Sweep: %d blocks, %d train / %d val proteins, lambda_phys=%.1f",
             NUM_BLOCKS, len(train_ids), len(val_ids), LAMBDA_PHYS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict] = []
    t_start = time.time()

    for b in range(NUM_BLOCKS):
        t0 = time.time()
        X_train, y_train_raw = load_block_data(b, train_ids)
        X_val, y_val_raw = load_block_data(b, val_ids)
        stats = FeatureStats.from_targets([y_train_raw])
        y_train_norm = stats.normalize(y_train_raw)
        y_val_norm = stats.normalize(y_val_raw)

        model = PSA(d_pair=128, d_hidden=128, K=K_PAIR)
        cfg = TrainConfig(epochs=EPOCHS, batch_size=8192, lr=1e-3, lambda_phys=LAMBDA_PHYS)
        history = train_psa(model, X_train, y_train_norm, X_val, y_val_norm, stats,
                             cfg=cfg, device=device)
        final = history[-1].copy()
        final["block"] = b
        final["sec"] = time.time() - t0
        results.append(final)

        # Save per-block model + stats so we can run inspection later without retraining.
        block_dir = RUNS_DIR / f"block_{b:02d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), block_dir / "model.pt")
        torch.save({"mean": stats.mean, "std": stats.std, "is_binary": stats.is_binary},
                    block_dir / "stats.pt")

        elapsed = time.time() - t_start
        log.info(
            "block %2d done in %.1fs (total %.0fs) | recon=%.3f d=%.2f c_acc=%.3f h_acc=%.3f orient=%.2f sep=%.2f",
            b, final["sec"], elapsed,
            final["r2_recon"], final["r2_distance"], final["acc_contact"],
            final["acc_hbond"], final["r2_orient_cos"], final["r2_log_seq_sep"],
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUNS_DIR / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    csv_cols = list(results[0].keys())
    with (RUNS_DIR / "sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        w.writerows(results)

    log.info("Wrote %s and .csv", summary_path)
    log.info("Total time: %.0fs", time.time() - t_start)


if __name__ == "__main__":
    main()
