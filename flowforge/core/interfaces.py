"""Protocols (structural interfaces) for FlowForge components.

Adhering to these lets the pipeline treat every model — numpy hand-written
flows, sklearn baselines, or optional ``nflows`` SOTA backends — uniformly.

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Flow(Protocol):
    """A normalizing flow: a bijective ``x <-> z`` map with tractable Jacobian.

    Convention: ``log_prob`` returns *higher = more likely* (log-density in
    nats).  ``forward`` maps data ``x`` to latent ``z``; ``inverse`` samples.
    """

    @property
    def n_params(self) -> int: ...

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x (N,D) -> (z (N,D), log_det (N,)) where log_det = log|det dx/dz|? .

        Returns ``log_det = log|det ∂z/∂x|`` (the encoder Jacobian log-det).
        """
        ...

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """z (N,D) -> x (N,D)."""
        ...

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Log-density of ``x`` under the model (N,) in nats."""
        ...

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` samples from the model."""
        ...


def available_nflows() -> bool:
    """True if the optional ``nflows`` + ``torch`` SOTA backend is importable."""
    try:
        import nflows  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False
