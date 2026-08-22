from __future__ import annotations

import torch
from torch import nn


class DistogramBottleneck(nn.Module):
    """Round-trip physical bottleneck: hidden -> distogram -> hidden."""

    def __init__(
        self,
        hidden_dim: int,
        bins: int = 38,
        bottleneck_dim: int = 128,
        max_sep: float = 32.0,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.max_sep = max_sep
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 2),
            nn.Linear(hidden_dim * 2 + 2, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bins),
        )
        self.edge_decoder = nn.Sequential(
            nn.LayerNorm(bins + 2),
            nn.Linear(bins + 2, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
        )
        self.token_decoder = nn.Sequential(
            nn.LayerNorm(bottleneck_dim * 2),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.encode(hidden)
        reconstructed = self.decode(logits)
        return logits, reconstructed

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = hidden.shape
        left = hidden[:, :, None, :].expand(batch_size, length, length, -1)
        right = hidden[:, None, :, :].expand(batch_size, length, length, -1)
        sep = self._separation_features(length, hidden.device, hidden.dtype)
        pair = torch.cat([left, right, sep.expand(batch_size, -1, -1, -1)], dim=-1)
        logits = self.encoder(pair)
        return 0.5 * (logits + logits.transpose(1, 2))

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, length, _, _ = logits.shape
        probs = torch.softmax(logits, dim=-1)
        sep = self._separation_features(length, logits.device, logits.dtype)
        edge_input = torch.cat([probs, sep.expand(batch_size, -1, -1, -1)], dim=-1)
        edge = self.edge_decoder(edge_input)

        eye = torch.eye(length, device=logits.device, dtype=torch.bool)
        mask = (~eye)[None, :, :, None].to(edge.dtype)
        denom = mask.sum(dim=2).clamp_min(1.0)
        outgoing = (edge * mask).sum(dim=2) / denom
        incoming = (edge.transpose(1, 2) * mask).sum(dim=2) / denom
        return self.token_decoder(torch.cat([outgoing, incoming], dim=-1))

    def _separation_features(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype)
        signed = (pos[None, :] - pos[:, None]) / self.max_sep
        abs_sep = signed.abs().clamp(max=1.0)
        return torch.stack([abs_sep, signed], dim=-1)[None]
