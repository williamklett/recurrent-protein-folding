"""Smoke test: load ESMFold v1 and fold a single short protein on GPU.

Validates the inference path before we wire up activation hooks.
Prints model load time, fold time, and a few summary stats from the output
(pLDDT, distogram shape) so we can confirm the model is working.

Usage:
    python scripts/fold_one.py
"""
from __future__ import annotations

import logging
import time

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("fold_one")

# Villin headpiece HP35 — 35 residues, well-folded mini-protein. Standard test case.
TEST_SEQ = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    log.info("Loading ESMFold v1 weights (one-time download ~6 GB)…")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    # HF-recommended mixed precision for ESMFold: ESM-2 LM in fp16, trunk + heads in fp32.
    # Full fp16 causes NaNs in the pTM head; full fp32 OOMs on a 24 GB card.
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    model.esm = model.esm.half()
    # ESMFold uses chunking to fit long sequences in memory. For short seqs we disable.
    model.trunk.set_chunk_size(64)
    log.info("Model loaded in %.1fs", time.time() - t0)
    log.info("Trunk has %d folding blocks", len(model.trunk.blocks))

    log.info("Tokenizing %d-residue sequence", len(TEST_SEQ))
    inputs = tokenizer([TEST_SEQ], return_tensors="pt", add_special_tokens=False).to(device)

    log.info("Running fold…")
    t0 = time.time()
    with torch.no_grad():
        out = model(inputs["input_ids"])
    log.info("Fold complete in %.1fs", time.time() - t0)

    log.info("Output keys: %s", list(out.keys()))
    if "positions" in out:
        log.info("Positions shape: %s", tuple(out["positions"].shape))
    if "plddt" in out:
        plddt = out["plddt"][0].float().cpu().numpy()
        # plddt is per-residue-per-atom; aggregate to per-residue mean.
        if plddt.ndim == 2:
            plddt = plddt.mean(axis=-1)
        log.info("Mean pLDDT: %.1f (≥ 70 = confident; ≥ 90 = very confident)", plddt.mean())
    if "distogram_logits" in out:
        log.info("Distogram logits shape: %s", tuple(out["distogram_logits"].shape))

    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1e9
        log.info("Peak GPU memory: %.2f GB", alloc)

    log.info("OK")


if __name__ == "__main__":
    main()
