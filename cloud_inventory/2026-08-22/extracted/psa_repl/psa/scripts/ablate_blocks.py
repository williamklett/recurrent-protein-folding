"""Ablate single ESMFold trunk blocks (set them to identity) and measure the
structural impact via distogram MSE / MAE and pLDDT.

If our "crystallization step" claim at block 42→43 is causal, ablating block 43
should hurt much more than ablating any other single block. We test a sweep of
ablation targets to see the full damage curve.

Method: forward hook on the target block that replaces its (s, z) output with
its (s, z) input — effectively skipping the block while letting the trunk's
4 recycle loop run normally.

Output: runs/ablation_summary.json — per (ablation_block × protein) row with
  - dist_mae_A: |predicted - true| Cα distance, mean over off-diagonal pairs
  - dist_mse_A2: squared version
  - plddt_mean: model's own confidence
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("ablate")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

# Ablation targets: baseline (no ablation) plus a sweep of blocks
ABLATION_BLOCKS = [None, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 42, 43, 44, 45, 46, 47]
SEED = 0


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    shuffled = pdb_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(frac_val * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def make_identity_hook():
    """Forward hook that replaces block output with block input.

    EsmFoldTriangularSelfAttentionBlock.forward(s, z, mask, ...) returns (s, z).
    We return the (s, z) input tensors instead.
    """
    def hook(module, args, output):
        # args[0] = s_in, args[1] = z_in
        s_in, z_in = args[0], args[1]
        return (s_in, z_in)
    return hook


def fold_protein(model, tok, seq: str) -> dict:
    ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.no_grad():
        out = model(ids)
    # out["positions"]: (n_recycle, batch, L, 14, 3) in nm
    positions = out["positions"][-1, 0]   # (L, 14, 3)
    ca = positions[:, 1, :]                # CA atom index = 1
    diff = ca[:, None, :] - ca[None, :, :]
    pred_dist = (diff ** 2).sum(-1).sqrt() * 10.0  # nm → Å
    plddt = out["plddt"][0].float().cpu().numpy()
    if plddt.ndim == 2:
        plddt = plddt.mean(axis=-1)
    return {"pred_dist": pred_dist.cpu().numpy(),
             "plddt_mean": float(plddt.mean())}


def evaluate_against_truth(pred_dist: np.ndarray, gt_dist: np.ndarray) -> dict:
    L = pred_dist.shape[0]
    eye = np.eye(L, dtype=bool)
    diff = (pred_dist - gt_dist)[~eye]
    return {
        "dist_mae_A": float(np.abs(diff).mean()),
        "dist_mse_A2": float((diff ** 2).mean()),
    }


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    log.info("Loading ESMFold (mixed precision)…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Loaded in %.1fs", time.time() - t0)

    # Load val protein sequences + ground-truth distances from physics
    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        seq = str(phys["sequence"])
        gt = phys["distances"].astype(np.float32)
        val_data[pid] = {"seq": seq, "gt_dist": gt, "L": gt.shape[0]}

    all_rows: list[dict] = []
    for ablate_block in ABLATION_BLOCKS:
        log.info("\n== Ablation: %s ==",
                  "BASELINE (no ablation)" if ablate_block is None else f"block {ablate_block}")
        handle = None
        if ablate_block is not None:
            handle = model.trunk.blocks[ablate_block].register_forward_hook(
                make_identity_hook()
            )
        try:
            for pid in val_ids:
                d = val_data[pid]
                if d["L"] != len(d["seq"]):
                    log.warning("  %s seq/gt length mismatch, skip", pid)
                    continue
                t0 = time.time()
                try:
                    pred = fold_protein(model, tok, d["seq"])
                except RuntimeError as e:
                    log.error("  %s fold failed: %s", pid, e)
                    torch.cuda.empty_cache()
                    continue
                if pred["pred_dist"].shape[0] != d["L"]:
                    log.warning("  %s pred shape %s vs gt L=%d",
                                pid, pred["pred_dist"].shape, d["L"])
                    continue
                metrics = evaluate_against_truth(pred["pred_dist"], d["gt_dist"])
                row = {
                    "ablate_block": "baseline" if ablate_block is None else ablate_block,
                    "pdb_id": pid,
                    "L": d["L"],
                    "fold_sec": time.time() - t0,
                    "plddt_mean": pred["plddt_mean"],
                    **metrics,
                }
                all_rows.append(row)
                torch.cuda.empty_cache()
            # Per-ablation summary
            sub = [r for r in all_rows if r["ablate_block"] ==
                   ("baseline" if ablate_block is None else ablate_block)]
            mae = np.mean([r["dist_mae_A"] for r in sub])
            mse = np.mean([r["dist_mse_A2"] for r in sub])
            plddt = np.mean([r["plddt_mean"] for r in sub])
            log.info("  summary | MAE=%.2f Å  MSE=%.2f Å²  pLDDT=%.3f  (n=%d proteins)",
                      mae, mse, plddt, len(sub))
        finally:
            if handle is not None:
                handle.remove()

    out_path = RUNS_DIR / "ablation_summary.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    log.info("Wrote %s with %d rows", out_path, len(all_rows))


if __name__ == "__main__":
    main()
