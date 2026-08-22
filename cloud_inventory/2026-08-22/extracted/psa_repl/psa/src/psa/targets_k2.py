"""K=2 target schema: pure geometry — distance and Cβ orientation cosine.

The hypothesis being tested: ESMFold's pair_rep encodes continuous geometric
primitives (distance, orientation), and the discrete features in our earlier
K=5 schema (contact, h-bond) were redundant because they are simply
thresholded views of these primitives.

If K=2 reaches reconstruction R² comparable to K=5 across the trunk, the
hypothesis is supported. If K=2 underperforms, the binary channels carried
non-redundant information we need to explain.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

PAIR_FEATURE_SCHEMA_K2: list[tuple[str, str]] = [
    ("distance", "reg"),
    ("orient_cos", "reg"),
]
K_PAIR_K2 = len(PAIR_FEATURE_SCHEMA_K2)
PAIR_FEATURE_NAMES_K2 = [name for name, _ in PAIR_FEATURE_SCHEMA_K2]


@dataclasses.dataclass
class FeatureStatsK2:
    """Pure-regression feature stats (no binary mask needed)."""

    mean: torch.Tensor   # (2,)
    std: torch.Tensor    # (2,)
    # Stub field so existing train/eval code that checks .is_binary still works.
    is_binary: torch.Tensor = dataclasses.field(
        default_factory=lambda: torch.tensor([False, False])
    )

    @classmethod
    def from_targets(cls, targets: list[torch.Tensor]) -> "FeatureStatsK2":
        all_y = torch.cat(targets, dim=0)
        mean = all_y.mean(dim=0)
        std = all_y.std(dim=0).clamp(min=1e-6)
        return cls(mean=mean, std=std, is_binary=torch.tensor([False, False]))

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean.to(y.device)) / self.std.to(y.device)

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std.to(y_norm.device) + self.mean.to(y_norm.device)


def build_pair_targets_k2(phys_npz_path: str | Path, mask_diagonal: bool = True
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 2) target tensor + (N,) keep mask for one protein.
    Columns: [distance (Å), orient_cos]."""
    d = np.load(phys_npz_path)
    distances = d["distances"]
    L = distances.shape[0]
    feats = np.stack(
        [
            distances.astype(np.float32).reshape(-1),
            d["orient_cos"].astype(np.float32).reshape(-1),
        ],
        axis=-1,
    )
    eye = np.eye(L, dtype=bool)
    keep = (~eye).reshape(-1) if mask_diagonal else np.ones(L * L, dtype=bool)
    return torch.from_numpy(feats), torch.from_numpy(keep)
