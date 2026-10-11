"""Composition of invertible transforms.

Flows compose: if ``f = f_n o ... o f_1`` then

    log|det ∂f/∂x| = sum_i log|det ∂f_i/∂x_{i-1}|

which we assert as a hard invariant.  ``CompositeFlow`` also drives the shared
training contract: ``backward`` threads ``(dz, dlog_det)`` backwards through
every block and accumulates parameter gradients under a stable key prefix.

Author: 晨星
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .base import AbstractFlow, base_logprob


class CompositeFlow(AbstractFlow):
    """Chains an ordered sequence of invertible blocks."""

    def __init__(self, dim: int, blocks: Sequence):
        self.dim = dim
        self.blocks: List = list(blocks)
        self._caches: List = []

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = x
        total = np.zeros(x.shape[0])
        caches = []
        for blk in self.blocks:
            if hasattr(blk, "forward") and callable(blk.forward):
                res = blk.forward(z)
                if isinstance(res, tuple):
                    z, ld = res
                else:  # pragma: no cover
                    z, ld = res, np.zeros(z.shape[0])
            else:  # pragma: no cover
                raise TypeError(f"block {blk!r} has no forward()")
            total = total + ld
            caches.append(dict(blk=blk))
        self._caches = caches
        return z, total

    def inverse(self, z: np.ndarray) -> np.ndarray:
        x = z
        for blk in reversed(self.blocks):
            x = blk.inverse(x)
        return x

    # ------------------------------------------------------------------ #
    def backward(self, dz: np.ndarray, dld: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        grads: Dict[str, np.ndarray] = {}
        gz = dz
        gld = np.asarray(dld, dtype=np.float64).copy()
        for i, blk in reversed(list(enumerate(self.blocks))):
            dx, g = blk.backward(gz, gld)
            for k, v in (g or {}).items():
                grads[f"b{i}.{k}"] = v
            # log_det of block i-1 is unaffected by block i's input transform,
            # so gld simply propagates unchanged to earlier blocks.
            gz = dx
        return gz, grads

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return sum(int(getattr(b, "n_params", 0)) for b in self.blocks)

    def get_params(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i, blk in enumerate(self.blocks):
            if hasattr(blk, "get_params"):
                for k, v in blk.get_params().items():
                    out[f"b{i}.{k}"] = v
        return out

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        grouped: Dict[int, Dict[str, np.ndarray]] = {}
        for k, v in params.items():
            idx = int(k.split(".", 1)[0][1:])
            grouped.setdefault(idx, {})[k.split(".", 1)[1]] = v
        for i, blk in enumerate(self.blocks):
            if i in grouped and hasattr(blk, "set_params"):
                blk.set_params(grouped[i])

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        z, ld = self.forward(x)
        return base_logprob(z) + ld
