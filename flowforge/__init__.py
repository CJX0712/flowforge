"""FlowForge — world-class normalizing-flow density estimation, in pure NumPy.

A deterministic, offline, CPU-only toolkit implementing RealNVP, MAF and the
FlowFuse mixture-of-flows flagship, with exact change-of-variables mathematics
and falsifiable invariants.

    from flowforge.pipeline.pipeline import FlowForgePipeline

Author: 晨星
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

from .core.seed import set_all  # noqa: F401  (re-exported convenience)

__all__ = ["__version__", "__author__", "set_all"]
