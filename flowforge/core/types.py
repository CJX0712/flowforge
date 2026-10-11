"""Core data types shared across FlowForge modules.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """A train/val/test split of a density-estimation problem.

    ``X`` is the training set; ``X_val`` is used only for model selection and
    early stopping (never for fitting scalers or HPO); ``X_test`` is the frozen
    holdout used for the final reported metric.
    """

    name: str
    dim: int
    X: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    def __post_init__(self) -> None:
        for attr in ("X", "X_val", "X_test"):
            arr = getattr(self, attr)
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float64:
                raise ValueError(f"{attr} must be float64 ndarray")
        if not (self.X.shape[1] == self.dim == self.X_val.shape[1] == self.X_test.shape[1]):
            raise ValueError("all splits must share dimension == dim")


@dataclass
class BenchmarkRow:
    """One row of the benchmark table (one model on one dataset)."""

    dataset: str
    model: str
    kind: str  # 'flagship' | 'flow' | 'baseline' | 'sota'
    nll_mean: float
    nll_std: float
    n_params: int
    skipped: bool = False
    note: str = ""

    @property
    def nll(self) -> float:
        return self.nll_mean


@dataclass
class BenchmarkReport:
    """Aggregated, reproducible benchmark result."""

    rows: list[BenchmarkRow] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, row: BenchmarkRow) -> None:
        self.rows.append(row)
