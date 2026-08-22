"""Causal same-rank signed-vs-PSD projection inside ESMFold trunk.

At selected trunk blocks, replace the pair representation z with a per-channel
low-rank projection over residue-pair axes:

  signed_abs:
      symmetrize each channel's LxL matrix, keep top-R |eigenvalue| components
      with their original signs.

  psd_abs_flip:
      keep the same top-R eigendirections, but force eigenvalues positive.
      This preserves rank/subspace/energy scale while removing indefinite sign.

  psd_pos:
      keep top-R positive eigencomponents only.

All three are the same rank budget. The critical contrast is signed_abs vs
psd_abs_flip: if middle blocks need indefinite signature, flipping the signs
should cause more damage there than preserving signed geometry.

Outputs:
  runs/signature_projection_intervention.json
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
log = logging.getLogger("sig_projection_intervention")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

SEED = 0
BLOCKS = [9, 25, 35, 43]
RANK = 16
MODES = ["signed_abs", "psd_abs_flip", "psd_pos"]


def split_proteins(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    s = pdb_ids.copy()
    rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def project_z_signature(z: torch.Tensor, rank: int, mode: str) -> torch.Tensor:
    """Project z with shape (1, L, L, C) channelwise over LxL matrices."""
    z0 = z[0].float()                         # (L, L, C)
    mats = z0.permute(2, 0, 1).contiguous()    # (C, L, L)
    mats = 0.5 * (mats + mats.transpose(-1, -2))

    vals, vecs = torch.linalg.eigh(mats)       # ascending, shapes (C,L), (C,L,L)
    C, L = vals.shape

    if mode in {"signed_abs", "psd_abs_flip"}:
        idx = torch.topk(vals.abs(), k=min(rank, L), dim=-1).indices
        lam = torch.gather(vals, 1, idx)
        if mode == "psd_abs_flip":
            lam = lam.abs()
    elif mode == "psd_pos":
        masked = vals.masked_fill(vals <= 0, float("-inf"))
        idx = torch.topk(masked, k=min(rank, L), dim=-1).indices
        lam = torch.gather(vals, 1, idx).clamp_min(0)
    else:
        raise ValueError(mode)

    gather_idx = idx[:, None, :].expand(C, L, idx.shape[1])
    U = torch.gather(vecs, 2, gather_idx)      # (C, L, R)
    recon = (U * lam[:, None, :]) @ U.transpose(-1, -2)
    recon = recon.permute(1, 2, 0).unsqueeze(0).to(dtype=z.dtype)
    return recon


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
    torch.manual_seed(SEED)
    random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

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

    conditions = [("baseline", None, None)] + [
        (f"{mode}_block{block}_rank{RANK}", mode, block)
        for block in BLOCKS
        for mode in MODES
    ]

    all_rows = []
    for condition, mode, block in conditions:
        log.info("\n== %s ==", condition)
        handle = None
        if mode is not None:
            handle = model.trunk.blocks[block].register_forward_hook(make_hook(RANK, mode))
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
                    "rank": RANK,
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

    out_path = RUNS_DIR / "signature_projection_intervention.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    log.info("Wrote %s with %d rows", out_path, len(all_rows))


if __name__ == "__main__":
    main()
