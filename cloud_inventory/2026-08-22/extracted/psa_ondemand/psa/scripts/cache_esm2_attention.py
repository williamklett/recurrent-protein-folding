"""Cache ESM-2 per-layer attention maps for every protein in the manifest.

For each protein, we run ESMFold's embedded ESM-2 backbone with
`output_attentions=True` and save a (num_layers, 1, 2*num_heads, L, L) tensor:

  axis 0 — layer index (0..35 for ESM-2 3B)
  axis 1 — batch (always 1)
  axis 2 — concatenated direction: first num_heads = forward attn[i,j],
           next num_heads = backward attn[j,i] (transposed)
  axis 3 — query (i)
  axis 4 — key (j)

We concat forward + backward attention into a single per-pair feature vector of
dimension 2*num_heads = 80, directly analogous to the 128-dim pair_rep we used
for ESMFold's trunk.

Output: data/esm2_attention/<pdb>.pt  with keys {pdb_id, L, attn} (fp16).
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("cache_esm2")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_DIR = ROOT / "data" / "esm2_attention"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    log.info("Manifest: %d proteins", len(rows))

    log.info("Loading ESMFold v1 (using its embedded ESM-2 for attention extraction)…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    esm2 = model.esm  # this is the EsmModel backbone (ESM-2 3B)
    log.info("Loaded in %.1fs. ESM-2 has %d layers, %d heads.",
              time.time() - t0,
              esm2.config.num_hidden_layers,
              esm2.config.num_attention_heads)

    n_cached, n_skipped = 0, 0
    for row in rows:
        pdb_id = row["pdb_id"]
        out_path = OUT_DIR / f"{pdb_id.lower()}.pt"
        if out_path.exists():
            n_skipped += 1
            continue

        seq = row["sequence"]
        L = len(seq)
        # ESM-2 tokenizer (same as ESMFold's) prepends CLS and appends EOS by default.
        # We want attention indexed by residue position, so suppress special tokens.
        ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()

        t0 = time.time()
        try:
            with torch.no_grad():
                out = esm2(
                    ids,
                    attention_mask=torch.ones_like(ids),
                    output_attentions=True,
                    return_dict=True,
                )
        except RuntimeError as e:
            log.error("%s  ESM-2 forward failed: %s", pdb_id, e)
            torch.cuda.empty_cache()
            continue

        # out.attentions is a tuple of length num_layers, each (B, H, L, L).
        # Stack to (num_layers, B, H, L, L), concat forward + transposed backward → 2H.
        attn = torch.stack(out.attentions, dim=0)        # (Layer, 1, H, L, L)
        attn_t = attn.transpose(-1, -2)                   # backward (j→i)
        attn_concat = torch.cat([attn, attn_t], dim=2)    # (Layer, 1, 2H, L, L)
        attn_concat = attn_concat.to(torch.float16).cpu()

        torch.save({"pdb_id": pdb_id, "L": L, "attn": attn_concat}, out_path)
        n_cached += 1
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        log.info("%-4s  L=%-4d  fwd=%.1fs  peak=%.1fGB  size=%.1fMB",
                  pdb_id, L, time.time() - t0, peak, out_path.stat().st_size / 1e6)
        torch.cuda.empty_cache()

    log.info("Done: %d cached, %d skipped (already present)", n_cached, n_skipped)


if __name__ == "__main__":
    main()
