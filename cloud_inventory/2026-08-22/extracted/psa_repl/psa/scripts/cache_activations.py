"""Run ESMFold on every protein in the manifest and cache per-block activations.

Writes one .pt file per protein under data/activations/, containing:
    pdb_id : str
    L      : int
    z      : (48, 1, L, L, 128) fp16   pair activations per block
    s      : (48, 1, L, 1024)   fp16   single activations per block

Idempotent: existing files are skipped. Designed to run on the GPU VM.

Usage:
    python scripts/cache_activations.py
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from psa.hooks import TrunkActivationCapture

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("cache_activations")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"


def main() -> None:
    ACT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    log.info("Manifest: %d proteins", len(rows))

    log.info("Loading ESMFold v1…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Model loaded in %.1fs", time.time() - t0)

    capture = TrunkActivationCapture(model)
    n_cached, n_skipped = 0, 0

    for row in rows:
        pdb_id = row["pdb_id"]
        out_path = ACT_DIR / f"{pdb_id.lower()}.pt"
        if out_path.exists():
            n_skipped += 1
            continue

        seq = row["sequence"]
        L = len(seq)
        ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

        t0 = time.time()
        try:
            with capture.capturing(), torch.no_grad():
                _ = model(ids)
        except RuntimeError as e:
            log.error("%s  FOLD FAILED: %s", pdb_id, e)
            torch.cuda.empty_cache()
            continue

        torch.save(
            {
                "pdb_id": pdb_id,
                "L": L,
                "z": capture.stacked_z(),
                "s": capture.stacked_s(),
            },
            out_path,
        )
        n_cached += 1
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        log.info("%-4s  L=%-4d  fold=%.1fs  peak=%.1fGB  size=%.1fMB",
                  pdb_id, L, time.time() - t0, peak, out_path.stat().st_size / 1e6)
        torch.cuda.empty_cache()

    log.info("Done: %d newly cached, %d already present", n_cached, n_skipped)


if __name__ == "__main__":
    main()
