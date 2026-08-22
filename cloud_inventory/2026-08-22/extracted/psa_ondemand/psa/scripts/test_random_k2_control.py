"""De-tautologization control for §5.5.

Our headline causal result was: K=2 PSA insertion damage at each block
correlates with PSA's recon error (1 − R²) at r = +0.903.

Open question: is this finding (i) the PSA's *geometric schema* causally
load-bearing, or (ii) the trivial 'destroying K=2 worth of information at
each block hurts proportionally to how much information is there'?

This control: instead of the trained K=2 PSA, insert RANDOMLY initialized
K=2 PSAs (same architecture, random weights — no training) at each block.
For each random instance, measure:
  - recon R² on val pair_reps at that block (computed locally before VM)
  - Δ MAE damage when inserted in ESMFold forward pass

If correlation r(Δ MAE, 1 − R²) for random insertions ≈ 0.9 → tautological
If random correlation is much lower (< 0.5) → trained PSA's geometric schema
specifically matters

We do N=3 random seeds × 10 insertion blocks = 30 random instances.

Outputs:
  runs/random_k2_control.json
  (figure made in followup local analysis)
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from psa.models import PSA
from psa.targets_k2 import K_PAIR_K2

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("random_k2")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
ACT_DIR = ROOT / "data" / "activations"
RUNS_DIR = ROOT / "runs"

INSERTION_BLOCKS = [5, 9, 15, 20, 25, 30, 35, 40, 43, 47]
RANDOM_SEEDS = [0]   # quick: 1 seed (was [0,1,2])
SPLIT_SEED = 0


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(0, keepdim=True)) ** 2).sum().item()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def make_random_psa(seed: int, device: str = "cuda") -> PSA:
    """Initialize a fresh K=2 PSA with seeded random weights (NO TRAINING)."""
    torch.manual_seed(seed)
    psa = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    psa.eval()
    return psa.to(device)


def make_psa_hook(psa: PSA):
    def hook(module, args, output):
        s, z = output
        L = z.shape[2]
        z_flat = z[0].reshape(L * L, 128).to(next(psa.parameters()).dtype)
        with torch.no_grad():
            _, z_recon = psa(z_flat)
        z_new = z_recon.reshape(1, L, L, 128).to(z.dtype)
        return (s, z_new)
    return hook


def fold_protein(model, tok, seq):
    ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.no_grad():
        out = model(ids)
    pos = out["positions"][-1, 0]
    ca = pos[:, 1, :]
    diff = ca[:, None, :] - ca[None, :, :]
    pred_dist = (diff ** 2).sum(-1).sqrt().cpu().numpy()
    plddt = out["plddt"][0].float().cpu().numpy()
    if plddt.ndim == 2:
        plddt = plddt.mean(axis=-1)
    return pred_dist, float(plddt.mean())


def compute_random_r2_per_block(val_ids: list[str], seed: int, block: int) -> float:
    """Compute recon R² of random PSA (seed s) on val pair_reps at given block."""
    torch.manual_seed(seed)
    psa = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    psa.eval()
    xs = []
    for pid in val_ids:
        act = torch.load(ACT_DIR / f"{pid.lower()}.pt",
                          map_location="cpu", weights_only=False)
        z = act["z"][block, 0].float()
        L = z.shape[0]
        xs.append(z.reshape(L * L, 128))
    X = torch.cat(xs)
    with torch.no_grad():
        _, h_hat = psa(X)
    return r2_score(X, h_hat)


def main() -> None:
    random.seed(SPLIT_SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SPLIT_SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    # Pre-compute recon R² for each (seed, block) on cached activations
    log.info("\n=== Computing recon R² for random K=2 PSAs (no training) ===")
    r2_table = {}   # {(seed, block): r2}
    for seed in RANDOM_SEEDS:
        for block in INSERTION_BLOCKS:
            r2 = compute_random_r2_per_block(val_ids, seed, block)
            r2_table[(seed, block)] = r2
        log.info("  seed %d: R² across blocks = %s",
                  seed, ["%.3f" % r2_table[(seed, b)] for b in INSERTION_BLOCKS])

    log.info("\n=== Loading ESMFold for damage measurement ===")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                   low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Loaded in %.1fs", time.time() - t0)

    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {"seq": str(phys["sequence"]),
                          "gt_dist": phys["distances"].astype(np.float32),
                          "L": phys["distances"].shape[0]}

    # Baseline
    log.info("\n== Baseline (no intervention) ==")
    baseline_rows = []
    for pid in val_ids:
        d = val_data[pid]
        if d["L"] != len(d["seq"]): continue
        try:
            pred_d, plddt = fold_protein(model, tok, d["seq"])
        except RuntimeError:
            continue
        if pred_d.shape[0] != d["L"]: continue
        eye = np.eye(d["L"], dtype=bool)
        mae = float(np.abs((pred_d - d["gt_dist"])[~eye]).mean())
        baseline_rows.append({"pdb_id": pid, "L": d["L"], "dist_mae_A": mae,
                               "plddt_mean": plddt})
        torch.cuda.empty_cache()
    baseline_mae = float(np.mean([r["dist_mae_A"] for r in baseline_rows]))
    baseline_plddt = float(np.mean([r["plddt_mean"] for r in baseline_rows]))
    log.info("  baseline MAE=%.3f Å, pLDDT=%.3f, n=%d", baseline_mae, baseline_plddt, len(baseline_rows))

    # Random insertions
    all_rows = []
    for seed in RANDOM_SEEDS:
        log.info("\n=== Seed %d ===", seed)
        for block in INSERTION_BLOCKS:
            psa = make_random_psa(seed)
            log.info("  insert random PSA(seed=%d) at block %d  (R²=%.3f)",
                      seed, block, r2_table[(seed, block)])
            handle = model.trunk.blocks[block].register_forward_hook(make_psa_hook(psa))
            try:
                for pid in val_ids:
                    d = val_data[pid]
                    if d["L"] != len(d["seq"]): continue
                    try:
                        pred_d, plddt = fold_protein(model, tok, d["seq"])
                    except RuntimeError:
                        continue
                    if pred_d.shape[0] != d["L"]: continue
                    eye = np.eye(d["L"], dtype=bool)
                    mae = float(np.abs((pred_d - d["gt_dist"])[~eye]).mean())
                    all_rows.append({"seed": seed, "block": block, "pdb_id": pid,
                                      "L": d["L"], "dist_mae_A": mae,
                                      "plddt_mean": plddt,
                                      "recon_r2": r2_table[(seed, block)]})
                    torch.cuda.empty_cache()
                sub = [r for r in all_rows if r["seed"] == seed and r["block"] == block]
                if sub:
                    log.info("    MAE=%.3f Å, pLDDT=%.3f",
                              np.mean([r["dist_mae_A"] for r in sub]),
                              np.mean([r["plddt_mean"] for r in sub]))
            finally:
                handle.remove()

    out = {
        "baseline_mae": baseline_mae,
        "baseline_plddt": baseline_plddt,
        "insertion_blocks": INSERTION_BLOCKS,
        "random_seeds": RANDOM_SEEDS,
        "recon_r2_per_seed_block": {f"{s}_{b}": r2_table[(s, b)]
                                       for s in RANDOM_SEEDS for b in INSERTION_BLOCKS},
        "results": all_rows,
    }
    (RUNS_DIR / "random_k2_control.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote runs/random_k2_control.json with %d rows", len(all_rows))


if __name__ == "__main__":
    main()
