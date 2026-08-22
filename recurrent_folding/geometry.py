from __future__ import annotations

import torch


def pairwise_distances(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diffs = x[:, :, None, :] - x[:, None, :, :]
    return torch.sqrt(torch.sum(diffs * diffs, dim=-1) + eps)


def center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=1, keepdim=True)


def bond_lengths(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x[:, 1:] - x[:, :-1], dim=-1)


def kabsch_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = center(pred)
    target_centered = center(target)
    cov = pred_centered.transpose(-1, -2) @ target_centered
    u, _, vh = torch.linalg.svd(cov)
    det = torch.det(u @ vh)
    fix = torch.ones(pred.shape[0], 3, device=pred.device, dtype=pred.dtype)
    fix[:, -1] = torch.sign(det)
    rot = u @ torch.diag_embed(fix) @ vh
    return pred_centered @ rot


def rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    aligned = kabsch_align(pred, target)
    target_centered = center(target)
    return torch.sqrt(torch.mean((aligned - target_centered) ** 2, dim=(1, 2)) + 1e-8)

