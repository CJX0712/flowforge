"""Mixture-of-Gaussians baseline (the *strong* parametric reference).

Implemented with ``scikit-learn`` (a top-tier, MIT-licensed open-source
library) — we explicitly reuse world-class code rather than reinvent EM.  The
flow flagship must beat this on every non-Gaussian benchmark to claim S-grade.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError
from .base import AbstractFlow


class MixtureOfGaussians(AbstractFlow):
    """Full-covariance GMM via ``sklearn.mixture.GaussianMixture``."""

    def __init__(self, dim: int, n_components: int = 8, reg_covar: float = 1e-3, seed: int = 0):
        self.dim = dim
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.seed = seed
        self._model = None

    def fit(self, x: np.ndarray) -> "MixtureOfGaussians":
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("scikit-learn required for GMM baseline") from exc
        n = min(self.n_components, max(1, x.shape[0] // 5))
        self.n_components = int(n)
        self._model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            reg_covar=self.reg_covar,
            max_iter=300,
            n_init=2,
            random_state=self.seed,
        )
        self._model.fit(x)
        return self

    # forward/inverse are not needed for the baseline (log_prob & sample
    # are overridden directly); provided as identity for protocol completeness.
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x, np.zeros(x.shape[0])

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MixtureOfGaussians.fit() must be called first")
        return self._model.score_samples(x)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MixtureOfGaussians.fit() must be called first")
        x, _ = self._model.sample(n)
        return x

    @property
    def n_params(self) -> int:
        if self._model is None:
            return 0
        # weights (K-1 free) + means (K*d) + covariances (K*d*(d+1)/2)
        k, d = self.n_components, self.dim
        return (k - 1) + k * d + k * d * (d + 1) // 2
