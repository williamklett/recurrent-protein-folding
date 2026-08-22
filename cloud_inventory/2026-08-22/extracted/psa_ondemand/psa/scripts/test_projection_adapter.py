"""Test trained projection adapters by inserting them in the forward pass and
measuring downstream structure quality.

Six conditions per protein:
  baseline                     full trunk + structure module (reference)
  truncate_only                identity for blocks 36..47, no adapter
                                (lower bound — what happens if we just skip late trunk)
  K2_psa_insert                K=2 PSA at block 35, identity 36..47
                                (heavy compression — known to break the model)
  linear_adapter               trained linear adapter at block 35, identity 36..47
  mlp_small_adapter            trained MLP (d_hidden=128), identity 36..47
  mlp_med_adapter              trained MLP (d_hidden=256), identity 36..47

If MLP adapter achieves baseline-similar quality, we've shown that 12 trunk
blocks can be compressed into a single learned linear or near-linear layer.

Output: runs/projection_adapter_intervention.json
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
from torch import nn
from transformers import AutoTokenizer, EsmForProteinFolding

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("test_adapter")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

SOURCE_BLOCK = 35
TARGET_BLOCK = 47
SEED = 0


class LinearAdapter(nn.Module):
    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.proj(h)


class MLPAdapter(nn.Module):
    def __init__(self, d: int = 128, d_hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)


def split_proteins(pdb_ids, frac_val=0.2, seed=0):
    rng = random.Random(seed)
    s = pdb_ids.copy(); rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def make_adapter_hook(adapter: nn.Module):
    def hook(module, args, output):
        s, z = output
        L = z.shape[2]
        z_flat = z[0].reshape(L * L, 128).to(next(adapter.parameters()).dtype)
        with torch.no_grad():
            z_new = adapter(z_flat)
        z_new = z_new.reshape(1, L, L, 128).to(z.dtype)
        return (s, z_new)
    return hook


def make_identity_hook():
    def hook(module, args, output):
        return (args[0], args[1])
    return hook


def fold_protein(model, tok, seq):
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


def load_adapter(arch_name: str, source_block: int, target_block: int):
    save_dir = RUNS_DIR / f"projection_adapter_{arch_name}_b{source_block}to{target_block}"
    if not save_dir.exists():
        return None
    if arch_name == "linear":
        model = LinearAdapter()
    elif arch_name == "mlp_small":
        model = MLPAdapter(d_hidden=128)
    elif arch_name == "mlp_med":
        model = MLPAdapter(d_hidden=256)
    else:
        raise ValueError(arch_name)
    model.load_state_dict(torch.load(save_dir / "model.pt",
                                       map_location="cpu", weights_only=True))
    model.eval().cuda()
    return model


def load_k2_psa(block_idx: int):
    """Load existing K=2 PSA for comparison."""
    from psa.models import PSA
    from psa.targets_k2 import K_PAIR_K2
    block_dir = RUNS_DIR / f"k2_block_{block_idx:02d}"
    if not block_dir.exists():
        return None
    psa = PSA(d_pair=128, d_hidden=128, K=K_PAIR_K2)
    psa.load_state_dict(torch.load(block_dir / "model.pt",
                                     map_location="cpu", weights_only=True))
    psa.eval().cuda()
    # Wrap so it has the same interface (h → h')
    class PsaWrapper(nn.Module):
        def __init__(self, psa):
            super().__init__()
            self.psa = psa
        def forward(self, h):
            _, h_hat = self.psa(h)
            return h_hat
        def parameters(self):
            return self.psa.parameters()
    return PsaWrapper(psa)


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

    # Load adapters
    adapters = {
        "linear": load_adapter("linear", SOURCE_BLOCK, TARGET_BLOCK),
        "mlp_small": load_adapter("mlp_small", SOURCE_BLOCK, TARGET_BLOCK),
        "mlp_med": load_adapter("mlp_med", SOURCE_BLOCK, TARGET_BLOCK),
        "k2_psa": load_k2_psa(SOURCE_BLOCK),
    }
    for k, v in adapters.items():
        log.info("  %s adapter loaded: %s", k, "yes" if v is not None else "MISSING")

    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {"seq": str(phys["sequence"]),
                          "gt_dist": phys["distances"].astype(np.float32),
                          "L": phys["distances"].shape[0]}

    # Conditions
    conditions = ["baseline", "truncate_only"]
    for adapter_name in adapters:
        if adapters[adapter_name] is not None:
            conditions.append(f"adapter_{adapter_name}")

    all_rows = []
    for cond in conditions:
        log.info("\n== Condition: %s ==", cond)
        handles = []
        if cond != "baseline":
            # Always insert identity hooks on blocks 36..47 (truncate the late trunk)
            for b in range(SOURCE_BLOCK + 1, 48):
                handles.append(model.trunk.blocks[b].register_forward_hook(make_identity_hook()))
            # Insert adapter hook (if applicable) at SOURCE_BLOCK
            if cond.startswith("adapter_"):
                adapter_name = cond[len("adapter_"):]
                ad = adapters[adapter_name]
                handles.append(model.trunk.blocks[SOURCE_BLOCK].register_forward_hook(
                    make_adapter_hook(ad)))
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
            for h in handles:
                h.remove()

    out_path = RUNS_DIR / "projection_adapter_intervention.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    log.info("\nWrote %s with %d rows", out_path, len(all_rows))


if __name__ == "__main__":
    main()
