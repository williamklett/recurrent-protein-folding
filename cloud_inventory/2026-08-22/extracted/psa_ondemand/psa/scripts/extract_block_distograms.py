"""Apply ESMFold's distogram head to every block's pair_rep for every protein,
producing per-block predicted distance + entropy maps. Saves one compact file
per protein.

Output (per protein):
    data/distograms/<pdb>.pt
      pdb_id
      L
      pred_distance : (48, L, L) fp16    mean of softmax(distogram_head(z_b))
      pred_entropy  : (48, L, L) fp16    entropy of that distribution
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
from transformers import EsmForProteinFolding

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("extract_distograms")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
OUT_DIR = ROOT / "data" / "distograms"

DISTO_MIN, DISTO_MAX, DISTO_N = 2.3125, 21.6875, 64
NUM_BLOCKS = 48


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    log.info("Loading ESMFold (using only distogram head)…")
    t0 = time.time()
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    head = model.distogram_head
    log.info("Loaded in %.1fs", time.time() - t0)

    bin_edges = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_N + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).cuda()

    n_cached, n_skipped = 0, 0
    for row in rows:
        pdb_id = row["pdb_id"]
        out_path = OUT_DIR / f"{pdb_id.lower()}.pt"
        if out_path.exists():
            n_skipped += 1
            continue
        act_path = ACT_DIR / f"{pdb_id.lower()}.pt"
        if not act_path.exists():
            continue

        act = torch.load(act_path, map_location="cpu", weights_only=False)
        z_all = act["z"]                # (48, 1, L, L, 128) fp16
        L = z_all.shape[2]

        pred_d = torch.zeros((NUM_BLOCKS, L, L), dtype=torch.float16)
        pred_e = torch.zeros((NUM_BLOCKS, L, L), dtype=torch.float16)
        t0 = time.time()
        with torch.no_grad():
            for b in range(NUM_BLOCKS):
                z = z_all[b, 0].float().cuda()                  # (L, L, 128)
                logits = head(z)                                # (L, L, 64)
                logits = logits + logits.transpose(-2, -3)      # symmetrize like real forward
                probs = torch.softmax(logits, dim=-1)           # (L, L, 64)
                mean_d = (probs * bin_centers).sum(-1)          # (L, L)
                ent = -(probs * (probs + 1e-12).log()).sum(-1)  # (L, L)
                pred_d[b] = mean_d.to(torch.float16).cpu()
                pred_e[b] = ent.to(torch.float16).cpu()
                torch.cuda.empty_cache()

        torch.save({"pdb_id": pdb_id, "L": L, "pred_distance": pred_d,
                     "pred_entropy": pred_e}, out_path)
        n_cached += 1
        log.info("%-4s  L=%-4d  fwd=%.1fs  size=%.1fMB",
                  pdb_id, L, time.time() - t0, out_path.stat().st_size / 1e6)

    log.info("Done: %d cached, %d skipped", n_cached, n_skipped)


if __name__ == "__main__":
    main()
