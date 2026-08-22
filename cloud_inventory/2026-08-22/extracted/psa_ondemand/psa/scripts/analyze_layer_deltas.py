"""For each layer transition b → b+1, train a K=2 PSA on the activation DELTA
  Δh_b(i,j) = h_{b+1}(i,j) − h_b(i,j)

This directly asks: how much of what each layer adds is physical?

Output:
  runs/layer_deltas_summary.json   per-transition recon R² + per-feature R²
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
from psa.train import TrainConfig, train_psa, r2_score

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s")
log = logging.getLogger("deltas")
log.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

NUM_BLOCKS = 48
LAMBDA_PHYS = 100.0
EPOCHS = 20
SEED = 0
D_PAIR = 128


def split_by_protein(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def load_delta_data(b: int, pdb_ids):
    """Concatenate Δh = h_{b+1} - h_b and K=2 physics targets across proteins."""
    xs, ys = [], []
    for pdb_id in pdb_ids:
        act = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt",
                          map_location="cpu", weights_only=False)
        h0 = act["z"][b, 0].float()
        h1 = act["z"][b + 1, 0].float()
        delta = h1 - h0                      # (L, L, 128)
        L = delta.shape[0]
        y_flat, keep = build_pair_targets_k2(PHYS_DIR / f"{pdb_id.lower()}.npz")
        if int(y_flat.shape[0] ** 0.5) != L:
            continue
        xs.append(delta.reshape(L * L, D_PAIR)[keep])
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
    log.info("Delta sweep: %d transitions, %d train / %d val proteins",
             NUM_BLOCKS - 1, len(train_ids), len(val_ids))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict] = []
    t_start = time.time()

    for b in range(NUM_BLOCKS - 1):
        t0 = time.time()
        X_train, y_train_raw = load_delta_data(b, train_ids)
        X_val, y_val_raw = load_delta_data(b, val_ids)

        # Delta magnitude (gives us "how much is the layer doing?")
        delta_mag_train = float((X_train ** 2).sum(-1).sqrt().mean().item())
        delta_mag_val = float((X_val ** 2).sum(-1).sqrt().mean().item())

        stats = FeatureStatsK2.from_targets([y_train_raw])
        y_train_norm = stats.normalize(y_train_raw)
        y_val_norm = stats.normalize(y_val_raw)

        model = PSA(d_pair=D_PAIR, d_hidden=128, K=K_PAIR_K2)
        cfg = TrainConfig(epochs=EPOCHS, batch_size=8192, lr=1e-3,
                           lambda_phys=LAMBDA_PHYS)
        history = train_psa(model, X_train, y_train_norm, X_val, y_val_norm,
                             stats, cfg=cfg, device=device,
                             feature_names=PAIR_FEATURE_NAMES_K2)
        final = history[-1]
        model.eval().to(device)
        with torch.no_grad():
            z_pred, _ = model(X_val.to(device))
        z_pred_raw = z_pred.cpu() * stats.std + stats.mean
        per_feat_r2 = {
            f"r2_{name}_raw": r2_score(y_val_raw[:, k : k + 1],
                                         z_pred_raw[:, k : k + 1])
            for k, name in enumerate(PAIR_FEATURE_NAMES_K2)
        }

        row = {
            "transition": f"{b}->{b+1}",
            "b": b,
            "delta_magnitude_train": delta_mag_train,
            "delta_magnitude_val": delta_mag_val,
            "r2_recon_delta": final["r2_recon"],
            **per_feat_r2,
            "sec": time.time() - t0,
        }
        results.append(row)
        log.info("Δ %2d→%2d in %.1fs (total %.0fs) | |Δ|=%.2f  recon=%.3f  d_R²=%.3f  orient_R²=%.3f",
                  b, b + 1, time.time() - t0, time.time() - t_start,
                  delta_mag_train, final["r2_recon"],
                  row["r2_distance_raw"], row["r2_orient_cos_raw"])

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / "layer_deltas_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s.  Total %.0fs.", out_path, time.time() - t_start)


if __name__ == "__main__":
    main()
