"""Scaled causal same-rank signed-vs-PSD projection grid.

This is the replication version of test_signature_projection_intervention.py.
It supports multiple ranks, more blocks, and a capped held-out validation set.

Output:
  runs/signature_projection_grid.json
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("sig_projection_grid")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

MODES = ["signed_abs", "psd_abs_flip", "psd_pos"]


def split_proteins(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    s = pdb_ids.copy()
    rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def project_z_signature(z: torch.Tensor, rank: int, mode: str) -> torch.Tensor:
    z0 = z[0].float()
    mats = z0.permute(2, 0, 1).contiguous()
    mats = 0.5 * (mats + mats.transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(mats)
    C, L = vals.shape
    k = min(rank, L)
    if mode in {"signed_abs", "psd_abs_flip"}:
        idx = torch.topk(vals.abs(), k=k, dim=-1).indices
        lam = torch.gather(vals, 1, idx)
        if mode == "psd_abs_flip":
            lam = lam.abs()
    elif mode == "psd_pos":
        masked = vals.masked_fill(vals <= 0, float("-inf"))
        idx = torch.topk(masked, k=k, dim=-1).indices
        lam = torch.gather(vals, 1, idx).clamp_min(0)
    else:
        raise ValueError(mode)
    gather_idx = idx[:, None, :].expand(C, L, k)
    U = torch.gather(vecs, 2, gather_idx)
    recon = (U * lam[:, None, :]) @ U.transpose(-1, -2)
    return recon.permute(1, 2, 0).unsqueeze(0).to(dtype=z.dtype)


def make_hook(rank: int, mode: str):
    def hook(module, args, output):
        s, z = output
        with torch.no_grad():
            z_new = project_z_signature(z, rank=rank, mode=mode)
        return (s, z_new)
    return hook


def fold_protein(model, tok, seq: str):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=str, default="5,9,13,17,21,25,29,33,35,39,43,47")
    ap.add_argument("--ranks", type=str, default="8,16,32")
    ap.add_argument("--max-val", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="signature_projection_grid.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [int(x) for x in args.blocks.split(",") if x]
    ranks = [int(x) for x in args.ranks.split(",") if x]

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows if (PHYS_DIR / f"{r['pdb_id'].lower()}.npz").exists()]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=args.seed)
    val_ids = val_ids[:args.max_val]
    log.info("Val proteins (%d): %s", len(val_ids), val_ids[:10])

    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {
            "seq": str(phys["sequence"]),
            "gt_dist": phys["distances"].astype(np.float32),
            "L": phys["distances"].shape[0],
        }

    log.info("Loading ESMFold...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Loaded in %.1fs", time.time() - t0)

    conditions = [("baseline", None, None, None)]
    for rank in ranks:
        for block in blocks:
            for mode in MODES:
                conditions.append((f"{mode}_block{block}_rank{rank}", mode, block, rank))

    all_rows = []
    for condition, mode, block, rank in conditions:
        log.info("\n== %s ==", condition)
        handle = None
        if mode is not None:
            handle = model.trunk.blocks[block].register_forward_hook(make_hook(rank, mode))
        try:
            for pid in val_ids:
                d = val_data[pid]
                if d["L"] != len(d["seq"]):
                    continue
                t0 = time.time()
                try:
                    pred_d, plddt = fold_protein(model, tok, d["seq"])
                except RuntimeError as e:
                    log.error("  %s failed: %s", pid, e)
                    torch.cuda.empty_cache()
                    continue
                if pred_d.shape[0] != d["L"]:
                    continue
                eye = np.eye(d["L"], dtype=bool)
                mae = float(np.abs((pred_d - d["gt_dist"])[~eye]).mean())
                all_rows.append({
                    "condition": condition,
                    "mode": mode or "baseline",
                    "block": block,
                    "rank": rank,
                    "pdb_id": pid,
                    "L": d["L"],
                    "dist_mae_A": mae,
                    "plddt_mean": plddt,
                    "fold_sec": time.time() - t0,
                })
                torch.cuda.empty_cache()
            sub = [r for r in all_rows if r["condition"] == condition]
            if sub:
                log.info("  summary | MAE=%.3f Å pLDDT=%.3f n=%d",
                         np.mean([r["dist_mae_A"] for r in sub]),
                         np.mean([r["plddt_mean"] for r in sub]),
                         len(sub))
        finally:
            if handle is not None:
                handle.remove()
        Path(RUNS_DIR / args.out).write_text(json.dumps(all_rows, indent=2))

    log.info("Wrote %s with %d rows", RUNS_DIR / args.out, len(all_rows))


if __name__ == "__main__":
    main()
