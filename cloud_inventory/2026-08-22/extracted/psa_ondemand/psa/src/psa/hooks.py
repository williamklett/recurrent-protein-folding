"""Forward-hook plumbing to capture per-block activations from the ESMFold trunk.

ESMFold's `trunk.blocks` is a `ModuleList` of 48 `EsmFoldTriangularSelfAttentionBlock`
modules. Each block returns the updated `(s, z)` pair where:
    s : (B, L, 1024)   single representation
    z : (B, L, L, 128) pair representation

We register a forward hook on each block, capture both tensors, and store them
in a dict keyed by block index. The model recycles 4 times — each block fires
once per recycle. We keep only the LAST firing (which is the final-recycle
output, i.e. what feeds the structure module).

Usage:
    capture = TrunkActivationCapture(model)
    with capture.capturing():
        model(input_ids)
    # capture.activations[i] = {'s': (1, L, 1024), 'z': (1, L, L, 128)}
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch


@dataclass
class BlockActivation:
    """Single block's outputs. Tensors live on CPU and are detached."""

    s: torch.Tensor  # (1, L, 1024)
    z: torch.Tensor  # (1, L, L, 128)


class TrunkActivationCapture:
    """Attach/detach forward hooks on every trunk block and collect their outputs.

    The hooks are installed when entering `capturing()` and removed on exit, so the
    model is unmodified between captures.
    """

    def __init__(self, model, store_dtype: torch.dtype = torch.float16) -> None:
        self.model = model
        self.store_dtype = store_dtype
        self.activations: dict[int, BlockActivation] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, block_idx: int):
        def hook(_module, _args, output):
            # Block output is (s, z) tuple.
            s, z = output
            self.activations[block_idx] = BlockActivation(
                s=s.detach().to(self.store_dtype).cpu(),
                z=z.detach().to(self.store_dtype).cpu(),
            )
        return hook

    def _install(self) -> None:
        blocks = self.model.trunk.blocks
        if len(self._handles) > 0:
            raise RuntimeError("hooks already installed")
        for i, block in enumerate(blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(i)))

    def _remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextlib.contextmanager
    def capturing(self):
        self.activations.clear()
        self._install()
        try:
            yield self
        finally:
            self._remove()

    @property
    def num_blocks(self) -> int:
        return len(self.activations)

    def stacked_z(self) -> torch.Tensor:
        """Return (num_blocks, 1, L, L, 128) stacked pair activations."""
        return torch.stack([self.activations[i].z for i in sorted(self.activations)], dim=0)

    def stacked_s(self) -> torch.Tensor:
        """Return (num_blocks, 1, L, 1024) stacked single activations."""
        return torch.stack([self.activations[i].s for i in sorted(self.activations)], dim=0)
