"""Gaussian baselines: isotropic and full-covariance.

These are the *weak* (isotropic) and *strong* (full-cov) parametric baselines
against which the flow flagship is measured.  They are maximum-likelihood
closed-form fits (no iteration), so they are fast, deterministic and a fair
reference: a genuine density-estimation SOTA must beat full-covariance Gaussian
on non-Gaussian data.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.errors import NumericalError
from .base import AbstractFlow


class IsotropicGaussian(AbstractFlow):
    """Spherical Gaussian baseline (single global variance)."""

    def __init__(self, dim: int, reg: float = 1e-6):
        self.dim = dim
        self.reg = reg
        self.mean = np.zeros(dim)
        self.var = 1.0

    def fit(self, x: np.ndarray) -> "IsotropicGaussian":
        self.mean = x.mean(axis=0)
        self.var = float(x.var(axis=0).mean()) + self.reg
        return self

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (x - self.mean) / np.sqrt(self.var)
        log_det = -0.5 * self.dim * np.log(self.var)  # log|det ∂z/∂x|
        return z, np.full(x.shape[0], log_det)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * np.sqrt(self.var) + self.mean

    @property
    def n_params(self) -> int:
        return self.dim + 1


class FullCovGaussian(AbstractFlow):
    """Full-covariance Gaussian (MVN) baseline via Cholesky-stable fit."""

    def __init__(self, dim: int, reg: float = 1e-4):
        self.dim = dim
        self.reg = reg
        self.mean = np.zeros(dim)
        self.L = np.eye(dim)  # Cholesky factor of precision-free covariance

    def fit(self, x: np.ndarray) -> "FullCovGaussian":
        self.mean = x.mean(axis=0)
        xc = x - self.mean
        cov = (xc.T @ xc) / max(len(x) - 1, 1)
        cov = cov + self.reg * np.eye(self.dim)
        # stable inverse via Cholesky of covariance
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            raise NumericalError(f"cov not PD: {exc}") from exc
        self.L = L  # store lower Cholesky of covariance
        return self

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xc = x - self.mean
        # z = L^{-1} xc  (whiten using covariance Cholesky)
        z = np.linalg.solve(self.L, xc.T).T
        # |det ∂z/∂x| = |det L^{-1}| = 1 / prod(diag L)
        log_det = -np.sum(np.log(np.diag(self.L)))
        return z, np.full(x.shape[0], log_det)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        xc = (self.L @ z.T).T
        return xc + self.mean

    @property
    def n_params(self) -> int:
        return self.dim + self.dim * (self.dim + 1) // 2
