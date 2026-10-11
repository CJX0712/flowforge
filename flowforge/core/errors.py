"""FlowForge error taxonomy (E100-E500).

Every exception carries a stable code so downstream tooling and CI can branch
on failure mode rather than parsing message text.

Author: 晨星
"""

from __future__ import annotations


class ForgeError(Exception):
    """Base class for all FlowForge errors (E100+)."""

    code = "E100"
    _registry: dict[str, type["ForgeError"]] = {}

    def __init__(self, message: str = "", *, code: str | None = None):
        self.message = message
        super().__init__(message)
        if code is not None:
            self.code = code

    def __init_subclass__(cls, **kwargs):  # noqa: D102
        super().__init_subclass__(**kwargs)
        if cls.code != "E100":
            ForgeError._registry[cls.code] = cls

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.code}] {self.message}"


class ConfigError(ForgeError):
    """E110: configuration / schema validation failure."""

    code = "E110"


class DataError(ForgeError):
    """E200: data loading / generation / leakage guard failure."""

    code = "E200"


class NumericalError(ForgeError):
    """E300: numerical instability (NaN / Inf / non-finite Jacobian)."""

    code = "E300"


class TrainingError(ForgeError):
    """E400: training loop failure (divergence / non-monotone loss)."""

    code = "E400"


class PipelineError(ForgeError):
    """E500: pipeline orchestration / I/O failure."""

    code = "E500"
