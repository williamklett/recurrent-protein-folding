"""Train a PSA on a real ESMFold trunk block and report R² metrics.

For a chosen block index, this:
  1. loads cached activations (z only — pair representation) for every protein
  2. loads physics targets for the same proteins
  3. flattens to (sum_p L_p², 128) inputs and (sum_p L_p², K) targets
  4. holds out 20% of proteins (NOT 20% of pairs — pairs within a protein are
     highly correlated; protein-level split is the principled choice)
  5. trains a small PSA, evaluates per-feature R²/accuracy on held-out proteins

Usage:
    python scripts/train_psa_real.py --block 47

Outputs:
    runs/block_<idx>/history.json
    runs/block_<idx>/model.pt
    runs/block_<idx>/stats.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import torch

from psa.models import PSA
from psa.targets import K_PAIR, FeatureStats, build_pair_targets
from psa.train import TrainConfig, evaluate, train_psa

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("train_psa_real")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"


def load_block_data(block_idx: int, pdb_ids: list[str]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate pair-rep activations and physics targets for a list of proteins.

    Returns:
      X : (N_total, 128) fp32   activations from block_idx
      y : (N_total, K)   fp32   raw (unnormalized) physical targets
    Diagonal pairs are masked out (we drop them).
    """
    xs, ys = [], []
    for pdb_id in pdb_ids:
        act = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        z = act["z"][block_idx, 0].float()  # (L, L, 128)
        L = z.shape[0]
        y_flat, keep = build_pair_targets(PHYS_DIR / f"{pdb_id.lower()}.npz")
        L_phys = int(y_flat.shape[0] ** 0.5)
        if L_phys != L:
            log.warning("%s  L mismatch (esm=%d phys=%d); skipping", pdb_id, L, L_phys)
            continue
        x_flat = z.reshape(L * L, 128)
        xs.append(x_flat[keep])
        ys.append(y_flat[keep])

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def split_by_protein(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0
                      ) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=47, help="trunk block index (0..47)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=K_PAIR)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    log.info("Found %d proteins with cached activations", len(pdb_ids))

    train_ids, val_ids = split_by_protein(pdb_ids, frac_val=0.2, seed=args.seed)
    log.info("Split: %d train / %d val proteins", len(train_ids), len(val_ids))

    log.info("Loading block %d activations + physics targets…", args.block)
    X_train, y_train_raw = load_block_data(args.block, train_ids)
    X_val, y_val_raw = load_block_data(args.block, val_ids)
    log.info("Train pairs: %d   Val pairs: %d", len(X_train), len(X_val))
    log.info("Activation stats — mean=%.3f std=%.3f", X_train.mean().item(), X_train.std().item())

    stats = FeatureStats.from_targets([y_train_raw])
    y_train_norm = stats.normalize(y_train_raw)
    y_val_norm = stats.normalize(y_val_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PSA(d_pair=128, d_hidden=args.d_hidden, K=args.K)
    log.info("PSA: %d params  device=%s", model.num_params, device)

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=8192, lr=args.lr,
        lambda_phys=args.lambda_phys,
    )
    history = train_psa(model, X_train, y_train_norm, X_val, y_val_norm, stats,
                         cfg=cfg, device=device)

    final = history[-1]
    log.info("Final block %d metrics:", args.block)
    for k, v in final.items():
        if k.startswith(("r2", "acc")):
            log.info("  %-15s %.4f", k, v)

    out_dir = RUNS_DIR / f"block_{args.block:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save({"mean": stats.mean, "std": stats.std, "is_binary": stats.is_binary},
                out_dir / "stats.pt")
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
