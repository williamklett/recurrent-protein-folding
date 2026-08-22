"""Build target tensors (z_phys ground truth) from cached physics .npz files.

The v0 bottleneck schema is intentionally tiny — 5 scalars per residue pair — so that
the experiment poses a hard question: *can ESMFold's 128-dim pair embedding be
reconstructed from only 5 physical scalars per pair?*

If yes, the model is doing physics. If no, what it's computing is richer than (or
orthogonal to) these specific physical quantities, and we expand the vocabulary.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

# Schema is intentionally hand-written so order is stable across training/eval.
# (name, kind) where kind ∈ {"reg", "bin"}.
PAIR_FEATURE_SCHEMA: list[tuple[str, str]] = [
    ("distance", "reg"),       # Cα–Cα distance, Å
    ("contact", "bin"),        # d_ij < 8 Å
    ("hbond", "bin"),          # Kabsch–Sander hbond
    ("log_seq_sep", "reg"),    # log(|i − j| + 1)
    ("orient_cos", "reg"),     # Cβ-vector dot product
]
K_PAIR = len(PAIR_FEATURE_SCHEMA)  # 5
PAIR_FEATURE_NAMES = [name for name, _ in PAIR_FEATURE_SCHEMA]


@dataclasses.dataclass
class FeatureStats:
    """Mean / std for normalizing regression features.

    Binary features get mean=0, std=1 (i.e. passthrough) — the is_binary mask
    tells the loss to use BCE on those columns instead of MSE.
    """

    mean: torch.Tensor      # (K,)
    std: torch.Tensor       # (K,)
    is_binary: torch.Tensor  # (K,) bool

    @classmethod
    def from_targets(cls, targets: list[torch.Tensor]) -> "FeatureStats":
        """Compute global statistics across the union of training targets."""
        all_y = torch.cat(targets, dim=0)  # (N_total, K)
        is_binary = torch.tensor(
            [kind == "bin" for _, kind in PAIR_FEATURE_SCHEMA], dtype=torch.bool
        )
        mean = all_y.mean(dim=0)
        std = all_y.std(dim=0)
        mean[is_binary] = 0.0
        std[is_binary] = 1.0
        std = std.clamp(min=1e-6)
        return cls(mean=mean, std=std, is_binary=is_binary)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean.to(y.device)) / self.std.to(y.device)

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)


def build_pair_targets(phys_npz_path: str | Path, mask_diagonal: bool = True
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct (N, K) target tensor + (N,) keep mask for one protein.

    N = L * L pairs. The keep mask flags non-diagonal pairs (and any other pair we
    don't want to train on, e.g. padding in the future).
    """
    d = np.load(phys_npz_path)
    distances = d["distances"]
    L = distances.shape[0]

    feats = np.stack(
        [
            distances.astype(np.float32).reshape(-1),
            d["contacts"].astype(np.float32).reshape(-1),
            d["hbonds"].astype(np.float32).reshape(-1),
            np.log1p(d["seq_sep"].astype(np.float32)).reshape(-1),
            d["orient_cos"].astype(np.float32).reshape(-1),
        ],
        axis=-1,
    )  # (L*L, K)

    eye = np.eye(L, dtype=bool)
    keep = (~eye).reshape(-1) if mask_diagonal else np.ones(L * L, dtype=bool)
    return torch.from_numpy(feats), torch.from_numpy(keep)
