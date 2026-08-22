"""Apply ESMFold's distogram head to every trunk block's pair_rep — "logit lens"
for protein folding. Asks the model's own distance decoder what each intermediate
block "thinks" the structure is.

For one protein, saves per-block predicted distance maps + entropy maps so we can
visualize the trajectory of internal predictions through the trunk.

Usage:
    python scripts/distogram_trajectory.py --pdb 1TIT
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import EsmForProteinFolding

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("distogram_trajectory")

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
OUT_DIR = ROOT / "runs"

# ESMFold/AF2 distogram convention: 64 bins from 2.3125 to 21.6875 Å.
DISTO_MIN, DISTO_MAX, DISTO_N = 2.3125, 21.6875, 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="1TIT")
    args = ap.parse_args()
    pdb_id = args.pdb.upper()

    log.info("Loading ESMFold (we'll only use distogram_head)…")
    t0 = time.time()
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    head = model.distogram_head
    log.info("Loaded in %.1fs", time.time() - t0)
    log.info("distogram_head: %s", head)

    n_bins = head.weight.shape[0]
    log.info("Head outputs %d bins; expecting %d", n_bins, DISTO_N)
    bin_edges = torch.linspace(DISTO_MIN, DISTO_MAX, DISTO_N + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).cuda()  # (64,)
    log.info("Bin centers: min=%.3f max=%.3f width=%.3f Å",
              bin_centers[0].item(), bin_centers[-1].item(),
              (bin_centers[1] - bin_centers[0]).item())

    log.info("Loading cached pair_rep for %s…", pdb_id)
    act = torch.load(ACT_DIR / f"{pdb_id.lower()}.pt", map_location="cpu", weights_only=False)
    z_all = act["z"]  # (48, 1, L, L, 128) fp16
    L = z_all.shape[2]
    n_blocks = z_all.shape[0]
    log.info("  L=%d, n_blocks=%d", L, n_blocks)

    gt = np.load(PHYS_DIR / f"{pdb_id.lower()}.npz")
    gt_distance = gt["distances"].astype(np.float32)
    log.info("  ground-truth distance range: %.2f to %.2f Å",
              gt_distance[gt_distance > 0].min(), gt_distance.max())

    pred_dist = np.zeros((n_blocks, L, L), dtype=np.float32)
    pred_entropy = np.zeros((n_blocks, L, L), dtype=np.float32)
    # Also keep the full distance distribution at every block, for a few pairs
    # we'll pick later — but storing all is 48*L*L*64*4 = ~70 MB for L=89, fine.
    pred_probs = np.zeros((n_blocks, L, L, n_bins), dtype=np.float16)

    with torch.no_grad():
        for b in range(n_blocks):
            z = z_all[b, 0].float().cuda()                     # (L, L, 128)
            logits = head(z)                                   # (L, L, 64)
            logits = logits + logits.transpose(-2, -3)         # symmetrize as in real forward
            probs = torch.softmax(logits, dim=-1)              # (L, L, 64)
            mean_d = (probs * bin_centers).sum(-1)             # (L, L)
            ent = -(probs * (probs + 1e-12).log()).sum(-1)     # (L, L)
            pred_dist[b] = mean_d.cpu().numpy()
            pred_entropy[b] = ent.cpu().numpy()
            pred_probs[b] = probs.to(torch.float16).cpu().numpy()
            torch.cuda.empty_cache()

    out_path = OUT_DIR / f"distogram_trajectory_{pdb_id.lower()}.pt"
    torch.save(
        {
            "pdb_id": pdb_id,
            "L": L,
            "pred_distance": pred_dist,    # (n_blocks, L, L)
            "pred_entropy": pred_entropy,  # (n_blocks, L, L)
            "pred_probs": pred_probs,      # (n_blocks, L, L, 64) fp16
            "gt_distance": gt_distance,    # (L, L)
            "bin_centers": bin_centers.cpu().numpy(),
        },
        out_path,
    )
    log.info("Saved %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Quick sanity readout
    final_err = np.abs(pred_dist[-1] - gt_distance).mean()
    block0_err = np.abs(pred_dist[0] - gt_distance).mean()
    log.info("Mean |pred_d − gt|:  block 0 = %.2f Å   block %d = %.2f Å",
              block0_err, n_blocks - 1, final_err)


if __name__ == "__main__":
    main()
