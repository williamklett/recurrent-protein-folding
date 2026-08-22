from __future__ import annotations

import torch
from torch.nn import functional as F

from recurrent_folding.geometry import bond_lengths, center, pairwise_distances, rmsd


DISTOGRAM_BINS = torch.linspace(2.0, 22.0, 37)


def physical_loss(trajectory: torch.Tensor, bond_target: float = 3.8) -> torch.Tensor:
    batch, steps, length, _ = trajectory.shape
    flat = trajectory.reshape(batch * steps, length, 3)
    bonds = bond_lengths(flat)
    bond_loss = torch.mean((bonds - bond_target) ** 2)

    dists = pairwise_distances(flat)
    sep = torch.arange(length, device=trajectory.device)
    nonlocal_mask = (sep[None, :] - sep[:, None]).abs() > 2
    clash = torch.relu(2.6 - dists[:, nonlocal_mask])
    clash_loss = torch.mean(clash**2)

    second_diff = flat[:, 2:] - 2 * flat[:, 1:-1] + flat[:, :-2]
    smooth_loss = torch.mean(second_diff**2)
    return bond_loss + 0.25 * clash_loss + 0.02 * smooth_loss


def reachability_loss(trajectory: torch.Tensor, max_step: float = 1.35) -> torch.Tensor:
    deltas = trajectory[:, 1:] - trajectory[:, :-1]
    jump = torch.linalg.norm(deltas, dim=-1)
    jump_loss = torch.mean(torch.relu(jump - max_step) ** 2)

    bond_delta = bond_lengths(deltas.reshape(-1, trajectory.shape[2], 3))
    coherent_motion = torch.mean(bond_delta**2)
    return jump_loss + 0.05 * coherent_motion


def reachability_from_start_loss(
    start: torch.Tensor,
    trajectory: torch.Tensor,
    max_step: float = 1.35,
) -> torch.Tensor:
    full = torch.cat([start[:, None], trajectory], dim=1)
    return reachability_loss(full, max_step=max_step)


def trajectory_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    coord = torch.mean((center(pred.reshape(-1, pred.shape[-2], 3)) - center(target.reshape(-1, target.shape[-2], 3))) ** 2)
    dist = torch.mean(
        (
            pairwise_distances(pred.reshape(-1, pred.shape[-2], 3))
            - pairwise_distances(target.reshape(-1, target.shape[-2], 3))
        )
        ** 2
    )
    return 0.3 * coord + 0.7 * dist


def ca_fape_lite_loss(pred: torch.Tensor, target: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
    """C-alpha local-frame loss inspired by AlphaFold FAPE.

    The toy stack only has C-alpha coordinates, so each residue frame is built
    from the local C-alpha tangent and a stable perpendicular axis.
    """
    pred_final = pred[:, -1] if pred.ndim == 4 else pred
    target_final = target[:, -1] if target.ndim == 4 else target
    pred_centered = center(pred_final)
    target_centered = center(target_final)
    pred_frames = _ca_frames(pred_centered)
    target_frames = _ca_frames(target_centered)

    pred_rel = pred_centered[:, None, :, :] - pred_centered[:, :, None, :]
    target_rel = target_centered[:, None, :, :] - target_centered[:, :, None, :]
    pred_local = torch.einsum("blij,blkj->blki", pred_frames.transpose(-1, -2), pred_rel)
    target_local = torch.einsum("blij,blkj->blki", target_frames.transpose(-1, -2), target_rel)
    error = torch.linalg.norm(pred_local - target_local, dim=-1)
    eye = torch.eye(error.shape[-1], device=error.device, dtype=torch.bool)[None]
    error = error[~eye.expand_as(error)]
    return torch.mean(torch.clamp(error, max=clamp) / clamp)


def distogram_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_final = target[:, -1] if target.ndim == 4 else target
    dists = pairwise_distances(target_final)
    bins = DISTOGRAM_BINS.to(device=target.device, dtype=target.dtype)
    labels = torch.bucketize(dists, bins)
    length = target_final.shape[1]
    mask = ~torch.eye(length, device=target.device, dtype=torch.bool)
    return F.cross_entropy(logits[:, mask, :].reshape(-1, logits.shape[-1]), labels[:, mask].reshape(-1))


def final_structure_loss(final: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    aligned = center(final)
    target_centered = center(target)
    coord = torch.mean((aligned - target_centered) ** 2)
    dist = torch.mean((pairwise_distances(final) - pairwise_distances(target)) ** 2)
    return 0.3 * coord + 0.7 * dist


def metrics(final: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            "rmsd": rmsd(final, target).mean().item(),
            "dist_mae": torch.mean(torch.abs(pairwise_distances(final) - pairwise_distances(target))).item(),
            "bond_mae": torch.mean(torch.abs(bond_lengths(final) - 3.8)).item(),
        }


def _ca_frames(coords: torch.Tensor) -> torch.Tensor:
    prev_idx = torch.arange(coords.shape[1], device=coords.device).sub(1).clamp_min(0)
    next_idx = torch.arange(coords.shape[1], device=coords.device).add(1).clamp_max(coords.shape[1] - 1)
    tangent = F.normalize(coords[:, next_idx] - coords[:, prev_idx], dim=-1, eps=1e-6)

    z_ref = torch.tensor([0.0, 0.0, 1.0], device=coords.device, dtype=coords.dtype).expand_as(tangent)
    y_ref = torch.tensor([0.0, 1.0, 0.0], device=coords.device, dtype=coords.dtype).expand_as(tangent)
    use_y = torch.abs((tangent * z_ref).sum(dim=-1, keepdim=True)) > 0.9
    ref = torch.where(use_y, y_ref, z_ref)
    normal = F.normalize(torch.cross(tangent, ref, dim=-1), dim=-1, eps=1e-6)
    binormal = F.normalize(torch.cross(tangent, normal, dim=-1), dim=-1, eps=1e-6)
    return torch.stack([tangent, normal, binormal], dim=-1)
