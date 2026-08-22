from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ESMFoldBlockActivation:
    s: torch.Tensor
    z: torch.Tensor


class ESMFoldTrunkCapture:
    """Capture final-recycle ESMFold trunk block activations."""

    def __init__(self, model, store_dtype: torch.dtype = torch.float16) -> None:
        self.model = model
        self.store_dtype = store_dtype
        self.activations: dict[int, ESMFoldBlockActivation] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, block_idx: int):
        def hook(_module, _args, output):
            s, z = output
            self.activations[block_idx] = ESMFoldBlockActivation(
                s=s.detach().to(self.store_dtype).cpu(),
                z=z.detach().to(self.store_dtype).cpu(),
            )

        return hook

    @contextlib.contextmanager
    def capturing(self):
        self.activations.clear()
        if self._handles:
            raise RuntimeError("ESMFold hooks already installed")
        for i, block in enumerate(self.model.trunk.blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(i)))
        try:
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

    def stacked_z(self) -> torch.Tensor:
        return torch.stack([self.activations[i].z for i in sorted(self.activations)], dim=0)


class PairDistogramBottleneck(nn.Module):
    """Pair activation bottleneck: z_ij -> distogram_ij -> reconstructed z_ij."""

    def __init__(
        self,
        pair_dim: int = 128,
        bins: int = 64,
        hidden_dim: int = 256,
        max_sep: float = 512.0,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.max_sep = max_sep
        self.encoder = nn.Sequential(
            nn.LayerNorm(pair_dim + 2),
            nn.Linear(pair_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bins),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(bins + 2),
            nn.Linear(bins + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pair_dim),
        )

    def forward(
        self,
        pair: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.encode(pair, i, j, length)
        reconstructed = self.decode(logits, i, j, length)
        return logits, reconstructed

    def encode(self, pair: torch.Tensor, i: torch.Tensor, j: torch.Tensor, length: int) -> torch.Tensor:
        features = torch.cat([pair, self._separation_features(i, j, length, pair.dtype)], dim=-1)
        return self.encoder(features)

    def decode(self, logits: torch.Tensor, i: torch.Tensor, j: torch.Tensor, length: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        features = torch.cat([probs, self._separation_features(i, j, length, logits.dtype)], dim=-1)
        return self.decoder(features)

    def _separation_features(
        self,
        i: torch.Tensor,
        j: torch.Tensor,
        length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        denom = float(max(length - 1, 1))
        signed = (j.to(dtype) - i.to(dtype)) / denom
        abs_sep = signed.abs().clamp(max=1.0)
        return torch.stack([abs_sep, signed], dim=-1)
