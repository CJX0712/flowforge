"""FlowForge core package: seed, config, errors, types, interfaces.

Author: 晨星
"""

from __future__ import annotations

from .config import Config
from .errors import (
    ConfigError,
    DataError,
    ForgeError,
    NumericalError,
    PipelineError,
    TrainingError,
)
from .seed import get_seed, set_all, spawn

__all__ = [
    "Config",
    "ConfigError",
    "DataError",
    "ForgeError",
    "NumericalError",
    "PipelineError",
    "TrainingError",
    "get_seed",
    "set_all",
    "spawn",
]
