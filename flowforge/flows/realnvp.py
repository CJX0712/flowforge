"""RealNVP assembly (Dinh et al., 2016) — alternating affine couplings.

Each block transforms one half of the coordinates; a random permutation (or an
alternating mask) between blocks guarantees every coordinate is transformed.
Composition keeps the map exactly invertible and the log-determinant exact.

Author: 晨星
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.seed import spawn
from .composite import CompositeFlow
from .coupling import AffineCoupling


def half_masks(dim: int) -> list[np.ndarray]:
    """Two complementary half-split masks (conditioning / transformed)."""
    a = np.ones(dim, dtype=np.float64)
    a[dim // 2 :] = 0.0
    b = 1.0 - a
    return [a, b]


class RealNVP(CompositeFlow):
    """Deep affine-coupling flow."""

    def __init__(
        self,
        dim: int,
        n_blocks: int = 4,
        hidden: int = 64,
        n_layers: int = 2,
        seed: int | None = None,
        name: str = "realnvp",
    ):
        masks = half_masks(dim)
        blocks = []
        rng = spawn(seed)
        base_seed = int(rng.integers(1 << 30))
        for b in range(n_blocks):
            mask = masks[b % 2]
            blk = AffineCoupling(
                dim=dim,
                mask=mask,
                hidden=hidden,
                n_layers=n_layers,
                seed=base_seed + b,
            )
            blocks.append(blk)
            blocks.append(RandomPermutation(dim, seed=base_seed + 1000 + b))
        super().__init__(dim, blocks)
        self.name = name


# Permutation lives in its own module; imported here for assembly convenience.
from .permute import RandomPermutation  # noqa: E402


def param_pointers(block) -> Dict[str, tuple]:
    """Resolve a block's gradient keys to mutable (container, subkey) refs."""
    out: Dict[str, tuple] = {}
    if isinstance(block, AffineCoupling):
        # single fused conditioner emitting [s | t]
        for k in block.net.params:
            out[k] = (block.net.params, k)
    elif hasattr(block, "made"):  # MAF
        for k in block.made.params:
            out[f"made.{k}"] = (block.made.params, k)
    return out
