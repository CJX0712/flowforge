"""Global deterministic seeding for FlowForge.

A single entry point ``set_all(seed)`` pins every RNG we use (numpy, python
``random``) so that two runs with the same seed produce bit-identical models
and metrics.  This is a hard contract of the project: the demo asserts it.

Author: 晨星
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

_GLOBAL_SEED: int = 12345


@dataclass(frozen=True)
class Rng:
    """A frozen view of the active global seed."""

    seed: int


def get_seed() -> int:
    return _GLOBAL_SEED


def set_all(seed: int = 12345) -> int:
    """Pin numpy + python ``random`` to ``seed`` and return it.

    Determinism contract: calling ``set_all(seed)`` then building/training a
    model must yield byte-identical parameters and metrics on every machine.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    np.random.seed(_GLOBAL_SEED)
    random.seed(_GLOBAL_SEED)
    return _GLOBAL_SEED


def spawn(seed: int | None = None) -> np.random.Generator:
    """Return a fresh ``numpy`` Generator derived from ``seed`` (or global)."""
    base = int(seed) if seed is not None else _GLOBAL_SEED
    return np.random.default_rng(base)
