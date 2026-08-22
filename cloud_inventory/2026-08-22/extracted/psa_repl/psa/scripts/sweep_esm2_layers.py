"""Sweep PSA training across every ESM-2 layer (36 layers, 80-dim per pair).

Directly analogous to scripts/sweep_all_blocks.py but for ESM-2 attention features
rather than ESMFold trunk pair_rep. Produces runs/esm2_sweep_summary.{json,csv}.
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
from psa.targets import K_PAIR, FeatureStats, build_pair_targets
from psa.train import TrainConfig, evaluate, train_psa

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")
log = logging.getLogger("sweep_esm2")
log.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ATTN_DIR = ROOT / "data" / "esm2_attention"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

LAMBDA_PHYS = 100.0
EPOCHS = 25
SEED = 0


def split_by_protein(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0
                      ) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_layer_data(layer_idx: int, pdb_ids: list[str]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ESM-2 attention features and physics targets for a list of proteins."""
    xs, ys = [], []
    for pdb_id in pdb_ids:
        cache = torch.load(ATTN_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
        # (Layer, 1, 2H, L, L)  →  layer slice: (1, 2H, L, L)  →  (L, L, 2H)
        attn = cache["attn"][layer_idx, 0].float()       # (2H, L, L)
        L = attn.shape[-1]
        attn = attn.permute(1, 2, 0).contiguous()        # (L, L, 2H)

        y_flat, keep = build_pair_targets(PHYS_DIR / f"{pdb_id.lower()}.npz")
        L_phys = int(y_flat.shape[0] ** 0.5)
        if L_phys != L:
            log.warning("%s  L mismatch (esm=%d phys=%d); skipping", pdb_id, L, L_phys)
            continue

        d_pair = attn.shape[-1]
        x_flat = attn.reshape(L * L, d_pair)
        xs.append(x_flat[keep])
        ys.append(y_flat[keep])

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ATTN_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    log.info("Found %d proteins with cached ESM-2 attention", len(pdb_ids))

    # Inspect feature dim by peeking at one protein.
    sample = torch.load(ATTN_DIR / f"{pdb_ids[0].lower()}.pt", map_location="cpu", weights_only=False)
    num_layers = sample["attn"].shape[0]
    d_pair = sample["attn"].shape[2]
    log.info("ESM-2 layers: %d, feature dim per pair: %d", num_layers, d_pair)

    train_ids, val_ids = split_by_protein(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Split: %d train / %d val proteins, lambda_phys=%.1f",
             len(train_ids), len(val_ids), LAMBDA_PHYS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict] = []
    t_start = time.time()

    for layer in range(num_layers):
        t0 = time.time()
        X_train, y_train_raw = load_layer_data(layer, train_ids)
        X_val, y_val_raw = load_layer_data(layer, val_ids)
        stats = FeatureStats.from_targets([y_train_raw])
        y_train_norm = stats.normalize(y_train_raw)
        y_val_norm = stats.normalize(y_val_raw)

        model = PSA(d_pair=d_pair, d_hidden=128, K=K_PAIR)
        cfg = TrainConfig(epochs=EPOCHS, batch_size=8192, lr=1e-3, lambda_phys=LAMBDA_PHYS)
        history = train_psa(model, X_train, y_train_norm, X_val, y_val_norm, stats,
                             cfg=cfg, device=device)
        final = history[-1].copy()
        final["layer"] = layer
        final["sec"] = time.time() - t0
        results.append(final)

        # Save per-layer model
        layer_dir = RUNS_DIR / f"esm2_layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), layer_dir / "model.pt")
        torch.save({"mean": stats.mean, "std": stats.std, "is_binary": stats.is_binary},
                    layer_dir / "stats.pt")

        log.info(
            "layer %2d in %.1fs (total %.0fs) | recon=%.3f d=%.2f c_acc=%.3f h_acc=%.3f orient=%.2f sep=%.2f",
            layer, final["sec"], time.time() - t_start,
            final["r2_recon"], final["r2_distance"], final["acc_contact"],
            final["acc_hbond"], final["r2_orient_cos"], final["r2_log_seq_sep"],
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with (RUNS_DIR / "esm2_sweep_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    cols = list(results[0].keys())
    with (RUNS_DIR / "esm2_sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    log.info("Wrote esm2_sweep_summary.{json,csv}. Total %.0fs.", time.time() - t_start)


if __name__ == "__main__":
    main()
