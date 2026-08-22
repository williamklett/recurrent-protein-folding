"""Fold a test protein with activation hooks attached. Confirms:
  * hooks fire for every one of the 48 trunk blocks
  * captured (s, z) shapes match expectations
  * we end up with the FINAL-recycle activations (not an intermediate recycle)
  * memory footprint of the full activation cache is reasonable

Usage:
    python scripts/fold_with_hooks.py
"""
from __future__ import annotations

import logging
import time

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from psa.hooks import TrunkActivationCapture

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("fold_with_hooks")

TEST_SEQ = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"  # villin headpiece HP35, 36 res


def main() -> None:
    device = "cuda"
    log.info("Loading ESMFold…")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    model.esm = model.esm.half()

    ids = tok([TEST_SEQ], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    L = ids.shape[1]
    log.info("Sequence length L = %d, trunk has %d blocks", L, len(model.trunk.blocks))

    capture = TrunkActivationCapture(model)
    t0 = time.time()
    with capture.capturing(), torch.no_grad():
        out = model(ids)
    dt = time.time() - t0
    log.info("Fold + capture in %.2fs", dt)

    # --- Invariant checks ---
    n = capture.num_blocks
    assert n == 48, f"expected 48 captured blocks, got {n}"
    log.info("Captured all 48 blocks ✓")

    last_s = capture.activations[47].s
    last_z = capture.activations[47].z
    assert tuple(last_s.shape) == (1, L, 1024), f"bad s shape {last_s.shape}"
    assert tuple(last_z.shape) == (1, L, L, 128), f"bad z shape {last_z.shape}"
    log.info("Final-block shapes: s=%s  z=%s ✓", tuple(last_s.shape), tuple(last_z.shape))

    # The last-block output should equal the model's reported final s_s / s_z
    # (modulo the dtype downcast we did in the hook).
    final_s = out["s_s"].detach().to(torch.float16).cpu()
    final_z = out["s_z"].detach().to(torch.float16).cpu()
    s_max_diff = (last_s - final_s).abs().max().item()
    z_max_diff = (last_z - final_z).abs().max().item()
    log.info("max|last_block_s − model.s_s| = %.6g", s_max_diff)
    log.info("max|last_block_z − model.s_z| = %.6g", z_max_diff)
    # Within fp16 quantization noise.
    assert s_max_diff < 0.05, "captured last-block s differs from reported s_s"
    assert z_max_diff < 0.05, "captured last-block z differs from reported s_z"
    log.info("Captured final activations match reported s_s / s_z ✓")

    # --- Memory footprint ---
    z_bytes = sum(a.z.element_size() * a.z.numel() for a in capture.activations.values())
    s_bytes = sum(a.s.element_size() * a.s.numel() for a in capture.activations.values())
    log.info(
        "Cache footprint @ L=%d, fp16: pair=%.1f MB single=%.1f MB total=%.1f MB",
        L, z_bytes / 1e6, s_bytes / 1e6, (z_bytes + s_bytes) / 1e6,
    )

    log.info("OK")


if __name__ == "__main__":
    main()
