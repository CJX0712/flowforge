"""Leak-safe standardisation.

A normalizing flow maps data to a *standard normal* base.  If the raw data has
mean 0.5 and std 1.7, the first thing the flow must learn is a trivial affine
rescaling — wasted capacity and wasted optimisation steps.  Standardising makes
the base prior a good match from step one and dramatically improves
convergence.

Leakage discipline (hard rule, enforced by tests)
------------------------------------------------
The scaler is fitted on **X_train only** and then applied unchanged to val and
test.  Because it is a fixed affine map, the test NLL shifts by the same
constant for *every* model — so cross-model comparison stays exactly fair.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import Dataset


class Standardizer:
    """Affine scaler fitted on the training split."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self.std = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer.fit() must be called first")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def log_abs_det(self) -> float:
        """log|det ∂z/∂x| of the mapping — constant across all models."""
        if self.std is None:
            return 0.0
        return float(-np.sum(np.log(self.std)))


def standardize_dataset(ds: Dataset) -> tuple[Dataset, Standardizer]:
    """Return a standardised copy of ``ds`` plus the train-fitted scaler."""
    sc = Standardizer().fit(ds.X)
    out = Dataset(
        name=ds.name,
        dim=ds.dim,
        X=sc.transform(ds.X),
        X_val=sc.transform(ds.X_val),
        X_test=sc.transform(ds.X_test),
    )
    return out, sc
