"""Physical Verbalizer (PV) and Physical Reconstructor (PR) modules.

Both are small MLPs operating on a single pair embedding h ∈ ℝ^d_pair at a time.
We do NOT use spatial context: each (i, j) pair is processed independently, which
makes the bottleneck a strict per-pair compression — a clean test of whether the
local pair embedding encodes physics.
"""

from __future__ import annotations

import torch
from torch import nn


class PhysicalVerbalizer(nn.Module):
    """h ∈ ℝ^d_pair  →  z_phys ∈ ℝ^K.

    z_phys has mixed semantics:
      * regression coords: in normalized space (matched to FeatureStats)
      * binary coords:     interpreted as logits (BCE applied in loss)
    """

    def __init__(self, d_pair: int = 128, d_hidden: int = 128, K: int = 5,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, K),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PhysicalReconstructor(nn.Module):
    """z_phys ∈ ℝ^K  →  ĥ ∈ ℝ^d_pair."""

    def __init__(self, K: int = 5, d_hidden: int = 128, d_pair: int = 128,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_pair),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PSA(nn.Module):
    """Physical System Autoencoder: h → z_phys → ĥ."""

    def __init__(self, d_pair: int = 128, d_hidden: int = 128, K: int = 5,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.pv = PhysicalVerbalizer(d_pair, d_hidden, K, dropout)
        self.pr = PhysicalReconstructor(K, d_hidden, d_pair, dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.pv(h)
        h_hat = self.pr(z)
        return z, h_hat

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
