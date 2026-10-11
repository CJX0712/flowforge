"""Configuration with environment overrides and schema validation.

Every tunable knob is read from ``ENV_FLOWFORGE_*`` if present, else falls back
to a validated default.  ``validate()`` is called at construction so a bad
config fails fast with an ``E110`` instead of a cryptic training crash.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import get_type_hints

from .errors import ConfigError

_PREFIX = "FLOWFORGE_"


@dataclass
class Config:
    """FlowForge runtime configuration (single source of truth)."""

    seed: int = 12345
    n_train: int = 2000
    n_val: int = 800
    n_test: int = 1500
    batch_size: int = 256
    epochs: int = 250
    lr: float = 8e-3
    weight_decay: float = 0.0
    hidden: int = 64
    n_layers: int = 2  # MLP depth inside each coupling/MADE conditioner
    n_coupling: int = 6  # affine-coupling blocks in RealNVP
    n_bins: int = 8  # spline bins (reserved: neural-spline coupling)
    bound: float = 5.0  # spline domain half-width
    patience: int = 60  # early-stop patience (val NLL)
    threads: int = 0  # 0 = numpy default

    def __post_init__(self) -> None:
        self._apply_env()
        self.validate()

    def _apply_env(self) -> None:
        # `from __future__ import annotations` makes Field.type a *string*, so
        # resolve real type objects before deciding how to coerce.
        hints = get_type_hints(type(self))
        for f in fields(self):
            env = f"{_PREFIX}{f.name.upper()}"
            if env not in os.environ:
                continue
            raw = os.environ[env]
            typ = hints.get(f.name, str)
            try:
                if typ is int:
                    setattr(self, f.name, int(raw))
                elif typ is float:
                    setattr(self, f.name, float(raw))
                elif typ is bool:
                    setattr(self, f.name, raw.strip().lower() in ("1", "true", "yes"))
                else:
                    setattr(self, f.name, raw)
            except ValueError as exc:
                raise ConfigError(f"bad env {env}={raw!r}: {exc}") from exc

    def validate(self) -> None:
        rules = {
            "seed": lambda v: v >= 0,
            "n_train": lambda v: v > 0,
            "n_val": lambda v: v > 0,
            "n_test": lambda v: v > 0,
            "batch_size": lambda v: 1 <= v <= 100_000,
            "epochs": lambda v: 0 < v <= 100_000,
            "lr": lambda v: 0 < v < 1.0,
            "weight_decay": lambda v: 0 <= v < 1.0,
            "hidden": lambda v: 1 <= v <= 4096,
            "n_layers": lambda v: 1 <= v <= 8,
            "n_coupling": lambda v: 1 <= v <= 64,
            "n_bins": lambda v: 2 <= v <= 64,
            "bound": lambda v: v > 0,
            "patience": lambda v: v >= 1,
            "threads": lambda v: v >= 0,
        }
        for name, ok in rules.items():
            val = getattr(self, name)
            if not ok(val):
                raise ConfigError(f"{name}={val!r} violates schema")

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ConfigError(f"unknown override {k}")
            setattr(cfg, k, v)
        cfg.validate()
        return cfg
