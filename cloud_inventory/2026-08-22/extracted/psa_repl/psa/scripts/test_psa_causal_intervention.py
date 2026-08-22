"""Insert the trained K=2 PSA at chosen trunk blocks as an inline adapter
during forward inference and measure downstream structure prediction quality.

Pipeline:
  At block b, the trunk normally outputs (s, z). We replace z with
    z' = PR(PV(z))
  where PV / PR are the K=2 PSA's encoder and decoder. The forward continues
  with z'.

If lift-and-project is the right algorithm and the trunk uses non-Euclidean
content in mid-blocks, this should:
  - degrade structure prediction substantially when inserted at mid blocks
    (where PSA recon R² is low — lots of information lost)
  - degrade negligibly when inserted at late blocks (where PSA recon R² is
    high — PSA close to identity)

Prediction: damage(b) ≈ 1 − recon_R²(b).
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
log = logging.getLogger("psa_causal")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

INSERTION_BLOCKS = [5, 9, 15, 20, 25, 30, 35, 40, 43, 47]
SEED = 0


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def load_psa(block_idx: int, device: str) -> PSA:
    block_dir = RUNS_DIR / f"k2_block_{block_idx:02d}"
    psa = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    psa.load_state_dict(torch.load(block_dir / "model.pt",
                                     map_location="cpu", weights_only=True))
    psa.eval()
    psa = psa.to(device)
    return psa


def make_psa_hook(psa: PSA):
    """Forward hook that replaces block output's pair_rep z with PR(PV(z))."""
    def hook(module, args, output):
        s, z = output
        # z shape: (B, L, L, 128); B=1 for inference
        L = z.shape[2]
        z_flat = z[0].reshape(L * L, 128).to(next(psa.parameters()).dtype)
        with torch.no_grad():
            _, z_recon = psa(z_flat)
        z_new = z_recon.reshape(1, L, L, 128).to(z.dtype)
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
    torch.manual_seed(SEED); random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows]
    _, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Val proteins (%d): %s", len(val_ids), val_ids)

    log.info("Loading ESMFold…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                   low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Loaded in %.1fs", time.time() - t0)

    # Load PSAs
    psas = {}
    for b in INSERTION_BLOCKS:
        psas[b] = load_psa(b, device="cuda")
        log.info("Loaded K=2 PSA for block %d", b)

    # Load val data
    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {"seq": str(phys["sequence"]),
                          "gt_dist": phys["distances"].astype(np.float32),
                          "L": phys["distances"].shape[0]}

    # Sweep conditions
    conditions = ["baseline"] + [f"insert_block{b}" for b in INSERTION_BLOCKS]
    all_rows = []
    for cond in conditions:
        log.info("\n== Condition: %s ==", cond)
        handle = None
        if cond.startswith("insert_block"):
            b = int(cond[len("insert_block"):])
            handle = model.trunk.blocks[b].register_forward_hook(make_psa_hook(psas[b]))
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
                all_rows.append({
                    "condition": cond, "pdb_id": pid, "L": d["L"],
                    "dist_mae_A": mae, "plddt_mean": plddt,
                    "fold_sec": time.time() - t0,
                })
                torch.cuda.empty_cache()
            sub = [r for r in all_rows if r["condition"] == cond]
            mae = np.mean([r["dist_mae_A"] for r in sub])
            plddt = np.mean([r["plddt_mean"] for r in sub])
            log.info("  summary | MAE=%.3f Å  pLDDT=%.3f  (n=%d)", mae, plddt, len(sub))
        finally:
            if handle is not None:
                handle.remove()

    out_path = RUNS_DIR / "psa_causal_intervention.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    log.info("\nWrote %s with %d rows", out_path, len(all_rows))


if __name__ == "__main__":
    main()
