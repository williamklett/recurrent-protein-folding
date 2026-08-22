#!/usr/bin/env python3
"""Locate and characterize the native ESMFold distogram readout + trunk dims."""
import torch
import torch.nn as nn
from transformers import EsmForProteinFolding

m = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
m.eval()

print("=== modules whose name mentions 'disto' or 'distance' ===")
for n, x in m.named_modules():
    if "disto" in n.lower() or "distance" in n.lower():
        np_ = sum(p.numel() for p in x.parameters(recurse=False))
        print(f"  {n:45s} {type(x).__name__:20s} own_params={np_}")
        for cn, c in x.named_children():
            print(f"      .{cn}: {c}")

print("\n=== top-level children of model ===")
for n, x in m.named_children():
    print(f"  {n}: {type(x).__name__}")

print("\n=== trunk structure ===")
print("num trunk blocks:", len(m.trunk.blocks))
# pair representation dim: peek a block
b0 = m.trunk.blocks[0]
print("block0 type:", type(b0).__name__)

# find the distogram head module object and print its forward signature-ish
cand = None
for n, x in m.named_modules():
    if n.endswith("distogram_head") or (("disto" in n.lower()) and isinstance(x, (nn.Linear, nn.Sequential))):
        cand = (n, x)
print("\nbest distogram-head candidate:", cand[0] if cand else None)
if cand:
    print(cand[1])
    for p_n, p in cand[1].named_parameters():
        print(f"   param {p_n}: {tuple(p.shape)}")

# how is distogram_logits produced? check forward source presence of attribute
print("\nhas attr distogram_head:", hasattr(m, "distogram_head"))
print("has attr trunk.distogram_head:", hasattr(m.trunk, "distogram_head"))
# config trunk dims
cfg = getattr(m.config, "esmfold_config", None)
if cfg is not None:
    print("\nesmfold_config.trunk:", getattr(cfg, "trunk", None))
