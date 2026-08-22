"""Training loop and loss for the Physical System Autoencoder.

The loss is

    L = ||h − ĥ||² + λ · L_phys

where L_phys is a mixed regression/classification loss on z_phys against the
ground-truth physical quantities. For regression features we use MSE in
normalized space; for binary features we use BCEWithLogits.

Both PV and PR are jointly optimized. Their MLPs are small (<1M params total),
so this trains in seconds per epoch on a single CPU for our starter set.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from psa.models import PSA
from psa.targets import FeatureStats

log = logging.getLogger(__name__)


def psa_loss(
    h: torch.Tensor,
    h_hat: torch.Tensor,
    z_pred: torch.Tensor,
    y_norm: torch.Tensor,
    stats: FeatureStats,
    lambda_phys: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined reconstruction + physical-supervision loss.

    Args:
      h        : (N, d_pair) ground-truth activations
      h_hat    : (N, d_pair) reconstructions
      z_pred   : (N, K) raw PV outputs — normalized space for reg, logits for bin
      y_norm   : (N, K) normalized targets (binary cols are 0/1 untouched)
      stats    : FeatureStats with is_binary mask
      lambda_phys : weight on the physical loss

    Returns:
      total_loss, dict of per-term scalars for logging.
    """
    is_binary = stats.is_binary.to(z_pred.device)
    is_reg = ~is_binary

    recon = F.mse_loss(h_hat, h)

    parts: list[torch.Tensor] = []
    if is_reg.any():
        parts.append(F.mse_loss(z_pred[:, is_reg], y_norm[:, is_reg]))
    if is_binary.any():
        parts.append(
            F.binary_cross_entropy_with_logits(z_pred[:, is_binary], y_norm[:, is_binary])
        )
    phys = torch.stack(parts).mean() if parts else torch.zeros((), device=h.device)

    total = recon + lambda_phys * phys
    return total, {"recon": recon.item(), "phys": phys.item(), "total": total.item()}


@torch.no_grad()
def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Coefficient of determination, averaged over output dimensions.

    For multi-dim outputs we compute SS_res and SS_tot per dimension then
    aggregate by summing both before the ratio — equivalent to "uniform_average"
    in sklearn with `multioutput='variance_weighted'`. We use sum-aggregate for
    a single scalar that's stable when some dims have tiny variance.
    """
    if y_true.numel() == 0:
        return float("nan")
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + eps))


@torch.no_grad()
def evaluate(model: PSA, X: torch.Tensor, y_norm: torch.Tensor,
              stats: FeatureStats, batch_size: int = 8192,
              feature_names: list[str] | None = None) -> dict[str, float]:
    """Return overall + per-feature reconstruction metrics on a held-out set.

    feature_names defaults to PAIR_FEATURE_NAMES (K=5 schema) for backward
    compatibility; pass an alternate list for other schemas (e.g. K=2).
    """
    model.eval()
    device = next(model.parameters()).device

    h_all, h_hat_all, z_pred_all = [], [], []
    for start in range(0, len(X), batch_size):
        x_batch = X[start : start + batch_size].to(device)
        z, h_hat = model(x_batch)
        h_all.append(x_batch.cpu())
        h_hat_all.append(h_hat.cpu())
        z_pred_all.append(z.cpu())
    h_all = torch.cat(h_all)
    h_hat_all = torch.cat(h_hat_all)
    z_pred_all = torch.cat(z_pred_all)

    metrics: dict[str, float] = {}
    metrics["r2_recon"] = r2_score(h_all, h_hat_all)

    if feature_names is None:
        from psa.targets import PAIR_FEATURE_NAMES
        feature_names = PAIR_FEATURE_NAMES

    is_bin = stats.is_binary
    for k, name in enumerate(feature_names):
        if k >= y_norm.shape[1]:
            break
        if is_bin[k]:
            preds = (z_pred_all[:, k] > 0).float()
            target = y_norm[:, k]
            acc = (preds == target).float().mean().item()
            metrics[f"acc_{name}"] = acc
        else:
            metrics[f"r2_{name}"] = r2_score(
                y_norm[:, k : k + 1], z_pred_all[:, k : k + 1]
            )

    model.train()
    return metrics


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_phys: float = 1.0
    log_every: int = 50


def make_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True
                  ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    n = len(X)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        sel = idx[start : start + batch_size]
        yield X[sel], y[sel]


def train_psa(
    model: PSA,
    X_train: torch.Tensor, y_train_norm: torch.Tensor,
    X_val: torch.Tensor, y_val_norm: torch.Tensor,
    stats: FeatureStats,
    cfg: TrainConfig = TrainConfig(),
    device: str = "cpu",
    feature_names: list[str] | None = None,
) -> list[dict[str, float]]:
    """Train PSA and return per-epoch validation metrics."""
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        step = 0
        running = {"recon": 0.0, "phys": 0.0, "total": 0.0, "n": 0}
        for x_batch, y_batch in make_batches(X_train, y_train_norm, cfg.batch_size):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z, h_hat = model(x_batch)
            loss, terms = psa_loss(x_batch, h_hat, z, y_batch, stats, cfg.lambda_phys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k in ("recon", "phys", "total"):
                running[k] += terms[k] * len(x_batch)
            running["n"] += len(x_batch)
            step += 1

        train_avg = {k: running[k] / running["n"] for k in ("recon", "phys", "total")}
        val = evaluate(model, X_val, y_val_norm, stats, batch_size=cfg.batch_size,
                        feature_names=feature_names)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_avg.items()}, **val}
        history.append(row)
        log.info(
            "epoch %2d | train recon=%.4f phys=%.4f | val r2_recon=%.4f "
            "r2_d=%.3f acc_c=%.3f acc_h=%.3f",
            epoch, train_avg["recon"], train_avg["phys"], val["r2_recon"],
            val.get("r2_distance", float("nan")),
            val.get("acc_contact", float("nan")),
            val.get("acc_hbond", float("nan")),
        )

    return history
