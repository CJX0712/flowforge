"""Fixed random permutation transform (RealNVP's "re-shuffle" step).

Coupling layers only transform a subset of coordinates, so consecutive layers
must swap which coordinates are active — otherwise the untouched half never
changes.  A permutation has ``log|det| = 0`` and its inverse is exact.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.seed import spawn


class RandomPermutation:
    """Deterministic (seeded) coordinate permutation."""

    def __init__(self, dim: int, seed: int | None = None):
        self.dim = dim
        rng = spawn(seed)
        self.perm = rng.permutation(dim)
        self.inv = np.argsort(self.perm)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:, self.perm], np.zeros(x.shape[0])

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z[:, self.inv]

    def backward(self, dz: np.ndarray, dld: np.ndarray) -> tuple[np.ndarray, dict]:
        return dz[:, self.inv], {}

    @property
    def n_params(self) -> int:
        return 0
