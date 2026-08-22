"""Scaled phase-adapter capacity sweep.

Tests the prescriptive claim:
  boundary phases are compressible, the middle relaxation phase is not.

For each phase and capacity:
  1. train adapter z_source -> z_target on train proteins
  2. evaluate pair-rep R2 on held-out proteins
  3. insert adapter, skip intervening blocks, evaluate final folding quality

Output:
  runs/adapter_capacity_sweep.json
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("adapter_capacity")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
PHYS_DIR = ROOT / "data" / "physics"
RUNS_DIR = ROOT / "runs"

PHASES = {
    "generation": (5, 17),
    "relaxation": (22, 34),
    "projection": (35, 47),
}


class Adapter(nn.Module):
    def __init__(self, d: int = 128, hidden: int = 0) -> None:
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(d, d)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, d),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)


def split_proteins(pdb_ids: list[str], frac_val: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    s = pdb_ids.copy()
    rng.shuffle(s)
    n_val = max(1, int(frac_val * len(s)))
    return s[n_val:], s[:n_val]


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(0, keepdim=True)) ** 2).sum().item()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def load_pair_data(pdb_ids: list[str], source: int, target: int, max_pairs: int | None, seed: int):
    rng = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for pid in pdb_ids:
        act = torch.load(ACT_DIR / f"{pid.lower()}.pt", map_location="cpu", weights_only=False)
        z_src = act["z"][source, 0].float()
        z_tgt = act["z"][target, 0].float()
        L = z_src.shape[0]
        x = z_src.reshape(L * L, 128)
        y = z_tgt.reshape(L * L, 128)
        if max_pairs is not None and len(x) > max_pairs:
            idx = torch.randperm(len(x), generator=rng)[:max_pairs]
            x = x[idx]
            y = y[idx]
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def train_adapter(model, X_train, y_train, X_val, y_val, epochs: int, batch_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    history = []
    for epoch in range(epochs):
        idx = torch.randperm(len(X_train), device=device)
        loss_sum = 0.0
        for start in range(0, len(X_train), batch_size):
            sel = idx[start:start + batch_size]
            pred = model(X_train[sel])
            loss = ((pred - y_train[sel]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(sel)
        with torch.no_grad():
            pred_val = model(X_val)
            val_mse = ((pred_val - y_val) ** 2).mean().item()
            val_r2 = r2_score(y_val.cpu(), pred_val.cpu())
        history.append({"epoch": epoch, "train_mse": loss_sum / len(X_train),
                         "val_mse": val_mse, "val_r2": val_r2})
    return model, history


def make_adapter_hook(adapter):
    def hook(module, args, output):
        s, z = output
        L = z.shape[2]
        z_flat = z[0].reshape(L * L, 128).to(next(adapter.parameters()).dtype)
        with torch.no_grad():
            z_new = adapter(z_flat)
        return (s, z_new.reshape(1, L, L, 128).to(z.dtype))
    return hook


def make_identity_hook():
    def hook(module, args, output):
        return (args[0], args[1])
    return hook


def fold_protein(model, tok, seq: str):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-sizes", type=str, default="0,64,256,1024,4096")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-proteins", type=int, default=220)
    ap.add_argument("--max-val-proteins", type=int, default=60)
    ap.add_argument("--max-pairs-per-protein", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="adapter_capacity_sweep.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x]

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows
               if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()
               and (PHYS_DIR / f"{r['pdb_id'].lower()}.npz").exists()]
    train_ids, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=args.seed)
    train_ids = train_ids[:args.max_train_proteins]
    val_ids = val_ids[:args.max_val_proteins]
    log.info("Split used: %d train / %d val proteins", len(train_ids), len(val_ids))

    val_data = {}
    for pid in val_ids:
        phys = np.load(PHYS_DIR / f"{pid.lower()}.npz")
        val_data[pid] = {"seq": str(phys["sequence"]),
                          "gt_dist": phys["distances"].astype(np.float32),
                          "L": phys["distances"].shape[0]}

    adapters = {}
    train_summary = {}
    for phase, (src, tgt) in PHASES.items():
        log.info("\n=== Phase %s %d->%d ===", phase, src, tgt)
        X_tr, y_tr = load_pair_data(train_ids, src, tgt, args.max_pairs_per_protein, args.seed)
        X_va, y_va = load_pair_data(val_ids, src, tgt, args.max_pairs_per_protein, args.seed + 1)
        id_r2 = r2_score(y_va, X_va)
        log.info("data train=%s val=%s identity R2=%.3f", tuple(X_tr.shape), tuple(X_va.shape), id_r2)
        for hidden in hidden_sizes:
            model = Adapter(hidden=hidden)
            n_params = sum(p.numel() for p in model.parameters())
            name = f"{phase}_h{hidden}"
            log.info("  training %s params=%d", name, n_params)
            trained, hist = train_adapter(model, X_tr, y_tr, X_va, y_va,
                                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            adapters[name] = trained.eval()
            train_summary[name] = {
                "phase": phase, "source": src, "target": tgt, "hidden": hidden,
                "n_params": n_params, "identity_r2": id_r2,
                "final_val_r2": hist[-1]["val_r2"],
                "final_val_mse": hist[-1]["val_mse"],
            }
            log.info("    final val R2=%.3f", hist[-1]["val_r2"])
        del X_tr, y_tr, X_va, y_va
        torch.cuda.empty_cache()

    log.info("\nLoading ESMFold for intervention tests...")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()

    conditions = [("baseline", None, None)]
    for phase in PHASES:
        conditions.append((f"truncate_{phase}", phase, None))
    for name, summ in train_summary.items():
        conditions.append((f"adapter_{name}", summ["phase"], name))

    results = []
    for cond, phase, adapter_name in conditions:
        log.info("\n== %s ==", cond)
        handles = []
        if phase is not None:
            src, tgt = PHASES[phase]
            for b in range(src + 1, tgt + 1):
                handles.append(model.trunk.blocks[b].register_forward_hook(make_identity_hook()))
            if adapter_name is not None:
                handles.append(model.trunk.blocks[src].register_forward_hook(
                    make_adapter_hook(adapters[adapter_name])))
        try:
            for pid in val_ids:
                d = val_data[pid]
                if d["L"] != len(d["seq"]):
                    continue
                t0 = time.time()
                try:
                    pred_d, plddt = fold_protein(model, tok, d["seq"])
                except RuntimeError as e:
                    log.error("  %s failed: %s", pid, e)
                    torch.cuda.empty_cache()
                    continue
                if pred_d.shape[0] != d["L"]:
                    continue
                eye = np.eye(d["L"], dtype=bool)
                mae = float(np.abs((pred_d - d["gt_dist"])[~eye]).mean())
                results.append({"condition": cond, "phase": phase or "baseline",
                                "adapter": adapter_name, "pdb_id": pid, "L": d["L"],
                                "dist_mae_A": mae, "plddt_mean": plddt,
                                "fold_sec": time.time() - t0})
                torch.cuda.empty_cache()
            sub = [r for r in results if r["condition"] == cond]
            log.info("  summary | MAE=%.3f Å pLDDT=%.3f n=%d",
                     np.mean([r["dist_mae_A"] for r in sub]),
                     np.mean([r["plddt_mean"] for r in sub]), len(sub))
        finally:
            for h in handles:
                h.remove()
        (RUNS_DIR / args.out).write_text(json.dumps({
            "phases": PHASES,
            "hidden_sizes": hidden_sizes,
            "train_summary": train_summary,
            "results": results,
        }, indent=2))

    log.info("Wrote %s", RUNS_DIR / args.out)


if __name__ == "__main__":
    main()
