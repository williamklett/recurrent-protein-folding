"""Phase-specificity control for the projection-adapter result.

Train adapters with the SAME architecture (MLP med, ~66K params) spanning
12-block ranges in three different phases of the trunk:

  generation   blocks  5 → 17   (candidate-construction phase)
  relaxation   blocks 22 → 34   (mid non-metric phase)
  projection   blocks 35 → 47   (re-projection phase; previously shown to compress)

Then for each phase, measure structure-prediction quality when the adapter
replaces the intervening 12 blocks during forward inference.

If the projection phase is uniquely compressible (because it implements a
single simple transformation, per the lift-and-project framing), the
projection-phase adapter should recover the bulk of the damage while the
other two should recover much less.

If all three phases compress similarly, the trunk has phase-agnostic block-
level redundancy and our 'projection is compressible' claim isn't specific
to projection — it's just true everywhere.

Outputs:
  runs/adapter_phase_specificity.json
  runs/adapter_phase_specificity.png
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
log = logging.getLogger("phase_specificity")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

SEED = 0
EPOCHS = 40
BATCH_SIZE = 8192
LR = 1e-3
D_HIDDEN = 256   # MLP_med architecture

PHASES = {
    "generation":  (5, 17),    # candidate construction
    "relaxation":  (22, 34),   # mid non-metric work
    "projection":  (35, 47),   # late re-projection (known-compressible)
}


class MLPAdapter(nn.Module):
    def __init__(self, d: int = 128, d_hidden: int = D_HIDDEN) -> None:
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


def load_pair_data(pdb_ids, source_block, target_block):
    xs, ys = [], []
    for pid in pdb_ids:
        act = torch.load(ACT_DIR / f"{pid.lower()}.pt",
                          map_location="cpu", weights_only=False)
        z_src = act["z"][source_block, 0].float()
        z_tgt = act["z"][target_block, 0].float()
        L = z_src.shape[0]
        xs.append(z_src.reshape(L * L, 128))
        ys.append(z_tgt.reshape(L * L, 128))
    return torch.cat(xs), torch.cat(ys)


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(0, keepdim=True)) ** 2).sum().item()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def train_adapter(phase_name, source, target, train_ids, val_ids, device="cuda"):
    X_tr, y_tr = load_pair_data(train_ids, source, target)
    X_va, y_va = load_pair_data(val_ids, source, target)
    log.info("  %s  src=%d→tgt=%d  train=%s val=%s",
              phase_name, source, target, X_tr.shape, X_va.shape)
    id_r2 = r2_score(y_va, X_va)
    model = MLPAdapter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_va = X_va.to(device); y_va = y_va.to(device)
    for epoch in range(EPOCHS):
        n = len(X_tr)
        idx = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            sel = idx[start:start+BATCH_SIZE]
            x, y = X_tr[sel], y_tr[sel]
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        val_r2 = r2_score(y_va.cpu(), model(X_va).cpu())
    log.info("    %s  identity R²=%.3f  trained val R²=%.3f", phase_name, id_r2, val_r2)
    return model, {"identity_r2": id_r2, "trained_val_r2": val_r2}


def make_adapter_hook(adapter):
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


def main() -> None:
    torch.manual_seed(SEED); random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    train_ids, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Split: %d train / %d val proteins", len(train_ids), len(val_ids))

    log.info("\n=== TRAIN adapters for each phase ===")
    adapters = {}
    train_summary = {}
    for phase_name, (src, tgt) in PHASES.items():
        adapter, summ = train_adapter(phase_name, src, tgt, train_ids, val_ids)
        adapters[phase_name] = adapter
        train_summary[phase_name] = {**summ, "source": src, "target": tgt}

    log.info("\n=== INFERENCE TEST ===")
    log.info("Loading ESMFold…")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                   low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    log.info("Loaded in %.1fs", time.time() - t0)

    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {"seq": str(phys["sequence"]),
                          "gt_dist": phys["distances"].astype(np.float32),
                          "L": phys["distances"].shape[0]}

    # Conditions: baseline + per-phase (truncate, adapter)
    conditions = ["baseline"]
    for phase_name in PHASES:
        conditions.append(f"truncate_{phase_name}")
        conditions.append(f"adapter_{phase_name}")

    all_rows = []
    for cond in conditions:
        log.info("\n== Condition: %s ==", cond)
        handles = []
        if cond.startswith("truncate_"):
            phase = cond[len("truncate_"):]
            src, tgt = PHASES[phase]
            for b in range(src + 1, tgt + 1):
                handles.append(model.trunk.blocks[b].register_forward_hook(make_identity_hook()))
        elif cond.startswith("adapter_"):
            phase = cond[len("adapter_"):]
            src, tgt = PHASES[phase]
            for b in range(src + 1, tgt + 1):
                handles.append(model.trunk.blocks[b].register_forward_hook(make_identity_hook()))
            handles.append(model.trunk.blocks[src].register_forward_hook(
                make_adapter_hook(adapters[phase])))
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
                all_rows.append({"condition": cond, "pdb_id": pid, "L": d["L"],
                                  "dist_mae_A": mae, "plddt_mean": plddt})
                torch.cuda.empty_cache()
            sub = [r for r in all_rows if r["condition"] == cond]
            mae = np.mean([r["dist_mae_A"] for r in sub])
            plddt = np.mean([r["plddt_mean"] for r in sub])
            log.info("  summary | MAE=%.3f Å  pLDDT=%.3f  (n=%d)", mae, plddt, len(sub))
        finally:
            for h in handles:
                h.remove()

    out = {"phases": PHASES, "train_summary": train_summary, "results": all_rows}
    (RUNS_DIR / "adapter_phase_specificity.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote runs/adapter_phase_specificity.json")


if __name__ == "__main__":
    main()
