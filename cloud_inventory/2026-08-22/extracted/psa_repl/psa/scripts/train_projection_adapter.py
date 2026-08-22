"""Train a small adapter that takes pair_rep at block SOURCE and predicts
pair_rep at block TARGET. If this adapter can substitute for the intervening
trunk blocks, we've shown that the projection phase is compressible into a
single learned layer — directly supporting the lift-and-project framing.

Trains three architectures:
  linear     : 128x128 matrix + residual (16K params)
  mlp_small  : 128 → 128 → 128 MLP + residual (33K params)
  mlp_med    : 128 → 256 → 128 MLP + residual (66K params)

Reports training/val MSE + R² against identity baseline (i.e., how much better
does the adapter do than just passing pair_rep through unchanged).

Output: runs/projection_adapter_{arch}/model.pt for each architecture.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("train_adapter")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
ACT_DIR = ROOT / "data" / "activations"
RUNS_DIR = ROOT / "runs"

SOURCE_BLOCK = 35
TARGET_BLOCK = 47
SEED = 0
EPOCHS = 40
BATCH_SIZE = 8192
LR = 1e-3


class LinearAdapter(nn.Module):
    """Linear residual: h + W h. Smallest possible (16K params)."""

    def __init__(self, d: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.proj(h)


class MLPAdapter(nn.Module):
    """Per-pair MLP residual."""

    def __init__(self, d: int = 128, d_hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d_hidden),
            nn.GELU(),
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
        z_src = act["z"][source_block, 0].float()      # (L, L, 128)
        z_tgt = act["z"][target_block, 0].float()
        L = z_src.shape[0]
        xs.append(z_src.reshape(L * L, 128))
        ys.append(z_tgt.reshape(L * L, 128))
    return torch.cat(xs), torch.cat(ys)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean(0, keepdim=True)) ** 2).sum().item()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def train_one(arch_name: str, model: nn.Module, X_train, y_train, X_val, y_val,
               device: str = "cuda"):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    X_train = X_train.to(device); y_train = y_train.to(device)
    X_val = X_val.to(device); y_val = y_val.to(device)

    # Identity baseline
    id_r2 = r2_score(y_val.cpu(), X_val.cpu())
    log.info("  identity baseline R²(X_val, y_val) = %.4f", id_r2)

    history = []
    for epoch in range(EPOCHS):
        n = len(X_train)
        idx = torch.randperm(n, device=device)
        running_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            sel = idx[start : start + BATCH_SIZE]
            x = X_train[sel]; y = y_train[sel]
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(x)
        with torch.no_grad():
            pred_val = model(X_val)
            val_mse = ((pred_val - y_val) ** 2).mean().item()
            val_r2 = r2_score(y_val.cpu(), pred_val.cpu())
        history.append({"epoch": epoch, "train_mse": running_loss / n,
                         "val_mse": val_mse, "val_r2": val_r2})
        if epoch in (0, 4, 9, 19, EPOCHS - 1):
            log.info("    epoch %2d  train_mse=%.4f  val_mse=%.4f  val_r2=%.4f",
                      epoch, running_loss / n, val_mse, val_r2)
    return model, history, id_r2


def main() -> None:
    torch.manual_seed(SEED); random.seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(MANIFEST)))
    pdb_ids = [r["pdb_id"] for r in rows
                if (ACT_DIR / f"{r['pdb_id'].lower()}.pt").exists()]
    train_ids, val_ids = split_proteins(pdb_ids, frac_val=0.2, seed=SEED)
    log.info("Split: %d train / %d val proteins", len(train_ids), len(val_ids))
    log.info("Source block: %d  Target block: %d", SOURCE_BLOCK, TARGET_BLOCK)

    X_train, y_train = load_pair_data(train_ids, SOURCE_BLOCK, TARGET_BLOCK)
    X_val, y_val = load_pair_data(val_ids, SOURCE_BLOCK, TARGET_BLOCK)
    log.info("Train: %s, Val: %s", X_train.shape, X_val.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    archs = {
        "linear":     LinearAdapter(),
        "mlp_small":  MLPAdapter(d_hidden=128),
        "mlp_med":    MLPAdapter(d_hidden=256),
    }
    summary = {}
    for arch_name, model in archs.items():
        n_params = sum(p.numel() for p in model.parameters())
        log.info("\n=== %s  (%d params) ===", arch_name, n_params)
        trained, history, id_r2 = train_one(arch_name, model,
                                              X_train, y_train, X_val, y_val,
                                              device=device)
        save_dir = RUNS_DIR / f"projection_adapter_{arch_name}_b{SOURCE_BLOCK}to{TARGET_BLOCK}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trained.state_dict(), save_dir / "model.pt")
        (save_dir / "history.json").write_text(json.dumps(history, indent=2))
        summary[arch_name] = {
            "n_params": n_params,
            "id_r2": id_r2,
            "final_val_mse": history[-1]["val_mse"],
            "final_val_r2": history[-1]["val_r2"],
            "save_dir": str(save_dir),
        }
        log.info("  saved to %s", save_dir)

    log.info("\n=== Summary ===")
    log.info("%-12s %8s %12s %12s %12s", "arch", "n_params", "id_R²", "val_MSE", "val_R²")
    for k, v in summary.items():
        log.info("%-12s %8d %12.4f %12.4f %12.4f",
                  k, v["n_params"], v["id_r2"], v["final_val_mse"], v["final_val_r2"])
    (RUNS_DIR / "projection_adapter_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
