"""Evaluation metrics for density models.

Primary metric: **test negative log-likelihood (NLL)** in nats — lower is
better, and it is directly comparable across every model family because all
models expose the same ``log_prob`` contract.

Secondary metric: **RBF-MMD²** between model samples and reference data — a
sample-quality signal that NLL alone cannot see (a model can assign high
likelihood yet generate poor samples).

Author: 晨星
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.errors import NumericalError
from ..core.seed import spawn


def negative_log_likelihood(model, x: np.ndarray) -> float:
    """Mean negative log-density (nats). Lower is better."""
    lp = np.asarray(model.log_prob(x), dtype=np.float64)
    if not np.all(np.isfinite(lp)):
        raise NumericalError("non-finite log-density during evaluation")
    return float(-np.mean(lp))


def bits_per_dim(model, x: np.ndarray) -> float:
    dim = x.shape[1]
    return negative_log_likelihood(model, x) / (dim * np.log(2.0))


def _rbf_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * sigma**2))


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Biased RBF-MMD² estimator between samples ``x`` and ``y``."""
    if sigma is None:
        # median heuristic on a subsample
        z = np.concatenate([x, y])
        m = min(len(z), 500)
        sub = z[:m]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        sigma = float(np.sqrt(med / 2.0)) if med > 0 else 1.0
        if sigma <= 0:
            sigma = 1.0
    kxx = _rbf_kernel(x, x, sigma)
    kyy = _rbf_kernel(y, y, sigma)
    kxy = _rbf_kernel(x, y, sigma)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def sample_mmd2(model, x_ref: np.ndarray, n: int = 500, seed: int = 0) -> float:
    """MMD² between ``n`` model samples and a reference sample (subsampled)."""
    rng = spawn(seed)
    ref = x_ref[:n]
    s = model.sample(n, rng=rng)
    m = min(n, len(ref))
    return rbf_mmd2(s[:m], ref[:m])


def aggregate(rows: Sequence[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` of a sequence of scores."""
    a = np.asarray(list(rows), dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))
