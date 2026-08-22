"""ReLU Sparse Autoencoder matching InterPLM's actual state_dict layout.

State dict keys (from `Elana/InterPLM-esm2-650m/layer_*/ae_unnormalized.pt`):
    bias            (d_in,)        applied as a pre-encoder / post-decoder shift
    encoder.weight  (d_dict, d_in)
    encoder.bias    (d_dict,)
    decoder.weight  (d_in, d_dict)  no decoder bias

Forward:
    z      = ReLU(encoder(x - bias))            sparse via L1 during training
    x_hat  = decoder(z) + bias

Sparsity is data-dependent (ReLU + L1 penalty during training induces it). We
do not apply TopK at inference — the model wasn't trained that way.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SAE(nn.Module):
    def __init__(self, d_in: int = 1280, d_dict: int = 10240) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_dict = d_dict
        self.bias = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, d_dict)
        self.decoder = nn.Linear(d_dict, d_in, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False):
        z = self.encode(x)
        x_hat = self.decode(z)
        return (x_hat, z) if output_features else x_hat


def load_interplm_sae(path: str | Path, d_in: int = 1280, d_dict: int = 10240
                       ) -> SAE:
    """Load an InterPLM `ae_unnormalized.pt` checkpoint into a SAE."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    sae = SAE(d_in=d_in, d_dict=d_dict)
    sae.load_state_dict(state, strict=True)
    sae.eval()
    return sae
