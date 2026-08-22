from __future__ import annotations

from dataclasses import dataclass

import torch


AA_COUNT = 20


@dataclass(frozen=True)
class ToyBatch:
    seq: torch.Tensor
    coords: torch.Tensor


def make_toy_batch(
    batch_size: int,
    length: int,
    device: torch.device,
    seed: int | None = None,
) -> ToyBatch:
    """Create deterministic protein-like backbone traces from sequence features.

    The synthetic fold mixes local amino-acid preferences with weak long-range
    attraction from hydrophobic residues. It is intentionally simple: the point
    is to validate recurrent, reachable folding dynamics before using PDB data.
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    seq = torch.randint(0, AA_COUNT, (batch_size, length), generator=gen, device=device)
    idx = torch.arange(length, device=device, dtype=torch.float32)[None, :]
    phase = seq.float().cumsum(dim=1) * 0.13

    hydrophobic = (seq % 5 == 0).float()
    polar = (seq % 7 == 0).float()
    turn = torch.sin(0.45 * idx + phase) + 0.35 * hydrophobic - 0.2 * polar
    rise = 0.35 + 0.08 * torch.cos(0.21 * idx + phase)

    x = torch.cos(turn.cumsum(dim=1))
    y = torch.sin(turn.cumsum(dim=1))
    z = rise
    steps = torch.stack([x, y, z], dim=-1)
    steps = steps / torch.linalg.norm(steps, dim=-1, keepdim=True).clamp_min(1e-6) * 3.8
    coords = torch.cumsum(steps, dim=1)

    # Sequence-informed compaction: a low-frequency bend plus hydrophobic collapse.
    hyd_center = (coords * hydrophobic[..., None]).sum(dim=1, keepdim=True)
    hyd_count = hydrophobic.sum(dim=1, keepdim=True).clamp_min(1.0)[..., None]
    hyd_center = hyd_center / hyd_count
    coords = coords - 0.18 * hydrophobic[..., None] * (coords - hyd_center)
    coords = coords + 1.2 * torch.sin(idx[..., None] / length * torch.pi) * torch.tensor(
        [0.4, -0.2, 0.1], device=device
    )
    coords = coords - coords.mean(dim=1, keepdim=True)
    return ToyBatch(seq=seq, coords=coords)

