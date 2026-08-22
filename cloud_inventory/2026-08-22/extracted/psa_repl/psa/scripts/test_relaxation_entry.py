"""Test the relaxation-entry prediction:
'Ablating blocks 23–25 together should damage structure more than any single
block, AND more than ablating triplets from other phases.'

Conditions (each ablation skips the listed blocks by replacing block.forward
output with input):
  baseline   no ablation
  b23         single 23
  b25         single 25
  b23-25      triplet [23, 24, 25]   ← relaxation entry (prediction)
  b14-16      triplet [14, 15, 16]   ← control: middle of metric phase
  b35-37      triplet [35, 36, 37]   ← control: middle of non-metric phase

Metric: distogram MAE vs ground truth, mean pLDDT, averaged across 10 val proteins.
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
log = logging.getLogger("relax_entry")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

CONDITIONS = {
    "baseline":  [],
    "b23":       [23],
    "b25":       [25],
    "b23-25":    [23, 24, 25],   # prediction: relaxation entry
    "b14-16":    [14, 15, 16],   # control: metric phase
    "b35-37":    [35, 36, 37],   # control: non-metric phase
}
SEED = 0


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def make_identity_hook():
    def hook(module, args, output):
        return (args[0], args[1])
    return hook


def fold_protein(model, tok, seq):
    ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.no_grad():
        out = model(ids)
    positions = out["positions"][-1, 0]
    ca = positions[:, 1, :]
    diff = ca[:, None, :] - ca[None, :, :]
    pred_dist = (diff ** 2).sum(-1).sqrt().cpu().numpy()    # Å (positions in Å)
    plddt = out["plddt"][0].float().cpu().numpy()
    if plddt.ndim == 2:
        plddt = plddt.mean(axis=-1)
    return pred_dist, float(plddt.mean())


def main():
    torch.manual_seed(SEED); random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    log.info("Loading ESMFold…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Loaded in %.1fs", time.time() - t0)

    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {"seq": str(phys["sequence"]),
                          "gt_dist": phys["distances"].astype(np.float32),
                          "L": phys["distances"].shape[0]}

    all_rows = []
    for cond, blocks in CONDITIONS.items():
        log.info("\n== Condition: %s  (blocks=%s) ==", cond, blocks)
        handles = []
        for b in blocks:
            handles.append(model.trunk.blocks[b].register_forward_hook(make_identity_hook()))
        try:
            for pid in val_ids:
                d = val_data[pid]
                if d["L"] != len(d["seq"]):
                    continue
                t0 = time.time()
                try:
                    pred_d, plddt = fold_protein(model, tok, d["seq"])
                except RuntimeError as e:
                    log.error("  %s fail: %s", pid, e)
                    torch.cuda.empty_cache(); continue
                if pred_d.shape[0] != d["L"]:
                    continue
                eye = np.eye(d["L"], dtype=bool)
                mae = float(np.abs((pred_d - d["gt_dist"])[~eye]).mean())
                mse = float(((pred_d - d["gt_dist"])[~eye] ** 2).mean())
                all_rows.append({
                    "condition": cond, "blocks": blocks,
                    "pdb_id": pid, "L": d["L"],
                    "dist_mae_A": mae, "dist_mse_A2": mse,
                    "plddt_mean": plddt, "fold_sec": time.time() - t0,
                })
                torch.cuda.empty_cache()
            sub = [r for r in all_rows if r["condition"] == cond]
            mae = np.mean([r["dist_mae_A"] for r in sub])
            plddt = np.mean([r["plddt_mean"] for r in sub])
            log.info("  summary | MAE=%.3f Å  pLDDT=%.3f  (n=%d)", mae, plddt, len(sub))
        finally:
            for h in handles:
                h.remove()

    out_path = RUNS_DIR / "relaxation_entry_test.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    log.info("Wrote %s with %d rows", out_path, len(all_rows))


if __name__ == "__main__":
    main()
