"""Cache per-residue ESM-2 650M activations + InterPLM SAE features for the
manifest proteins, at the 3 SAE layers we'll analyze in v1.

Output per protein:  data/esm2_650m_sae/<pdb>.pt
  pdb_id : str
  L      : int
  layers : [9, 24, 30]
  activations : {9: (L, 1280), 24: (L, 1280), 30: (L, 1280)}   fp16
  features    : {9: (L, 10240), 24: (L, 10240), 30: (L, 10240)} fp16, sparse

InterPLM was trained on activations after CLS/EOS stripping. We feed ESM-2 with
add_special_tokens=True (default), then slice off positions 0 and -1 before
applying the SAE — matches the training distribution.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from psa.sae import load_interplm_sae

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("cache_esm2_650m_sae")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
OUT_DIR = ROOT / "data" / "esm2_650m_sae"

ESM2_650M = "facebook/esm2_t33_650M_UR50D"
SAE_REPO = "Elana/InterPLM-esm2-650m"
LAYERS = [9, 24, 30]   # early / middle / late; all present in the InterPLM repo


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    log.info("Manifest: %d proteins", len(rows))

    log.info("Loading ESM-2 650M…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ESM2_650M)
    model = AutoModel.from_pretrained(ESM2_650M, torch_dtype=torch.float16)
    model = model.cuda().eval()
    log.info("Loaded ESM-2 650M in %.1fs (%d params)",
              time.time() - t0, sum(p.numel() for p in model.parameters()))

    log.info("Downloading InterPLM SAEs for layers %s…", LAYERS)
    saes: dict[int, torch.nn.Module] = {}
    for layer in LAYERS:
        path = hf_hub_download(repo_id=SAE_REPO, filename=f"layer_{layer}/ae_unnormalized.pt")
        sae = load_interplm_sae(path, d_in=1280, d_dict=10240)
        sae = sae.cuda().to(torch.float16)
        saes[layer] = sae
        log.info("  layer %d  d_dict=%d  d_in=%d", layer, sae.d_dict, sae.d_in)

    n_cached, n_skipped = 0, 0
    for row in rows:
        pdb_id = row["pdb_id"]
        out_path = OUT_DIR / f"{pdb_id.lower()}.pt"
        if out_path.exists():
            n_skipped += 1
            continue
        seq = row["sequence"]
        L = len(seq)

        ids = tok([seq], return_tensors="pt", add_special_tokens=True)
        input_ids = ids["input_ids"].cuda()
        attn_mask = ids["attention_mask"].cuda()

        t0 = time.time()
        try:
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        except RuntimeError as e:
            log.error("%s  ESM-2 forward failed: %s", pdb_id, e)
            torch.cuda.empty_cache()
            continue

        # out.hidden_states: tuple of (n_layers + 1,) of (B, L+2, 1280).
        # Index i corresponds to "after layer i" with hidden_states[0] being embeddings.
        # Strip CLS at 0 and EOS at -1 to get (L, 1280) per-residue.
        activations: dict[int, torch.Tensor] = {}
        features: dict[int, torch.Tensor] = {}
        for layer in LAYERS:
            h = out.hidden_states[layer][0, 1:-1, :]   # (L, 1280) fp16
            assert h.shape == (L, 1280), f"unexpected shape {h.shape} for {pdb_id}"
            with torch.no_grad():
                _, z = saes[layer](h, output_features=True)
            activations[layer] = h.detach().cpu()
            features[layer] = z.detach().cpu()
            torch.cuda.empty_cache()

        torch.save({"pdb_id": pdb_id, "L": L, "layers": LAYERS,
                     "activations": activations, "features": features}, out_path)
        n_cached += 1
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        log.info("%-4s  L=%-4d  fwd+sae=%.1fs  peak=%.1fGB  size=%.1fMB",
                  pdb_id, L, time.time() - t0, peak, out_path.stat().st_size / 1e6)

    log.info("Done: %d cached, %d skipped", n_cached, n_skipped)


if __name__ == "__main__":
    main()
