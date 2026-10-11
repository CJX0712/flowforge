"""Abstract flow base + standard-normal base distribution helpers.

Convention (fixed contract across the whole library):
  * ``forward(x)`` maps data ``x`` to latent ``z`` and returns
    ``(z, log_det)`` where ``log_det = log|det ∂z/∂x|``.
  * ``log_prob(x) = log p_base(z) + log_det``  (in nats, higher = denser).
  * ``inverse(z)`` is the generative map ``z -> x``.

Because every flow shares this contract, the pipeline can evaluate, sample and
train them uniformly — and cross-module metrics stay comparable.

Author: 晨星
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.seed import spawn


def base_logprob(z: np.ndarray) -> np.ndarray:
    """Log-density of the standard normal base distribution (nats)."""
    d = z.shape[1]
    return -0.5 * d * np.log(2.0 * np.pi) - 0.5 * np.sum(z**2, axis=1)


def sample_base(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, dim))


class AbstractFlow(ABC):
    """Common scaffolding: ``log_prob`` / ``sample`` derived from ``forward``/``inverse``."""

    dim: int

    @abstractmethod
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x (N,D) -> (z, log_det)."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, z: np.ndarray) -> np.ndarray:
        """z (N,D) -> x."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_params(self) -> int:
        raise NotImplementedError

    # ---- derived utilities ------------------------------------------- #
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        z, log_det = self.forward(x)
        return base_logprob(z) + log_det

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or spawn()
        z = sample_base(n, self.dim, rng)
        return self.inverse(z)

    def forward_kl_to(self, target_samples: np.ndarray) -> float:
        """Monte-Carlo forward KL ``KL(target || model)`` using target samples."""
        return float(-np.mean(self.log_prob(target_samples)))
