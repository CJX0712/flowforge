"""Pipeline package: orchestration, fusion flagship, CLI-facing API.

Author: 晨星
"""

from __future__ import annotations

from .flagship import FlowFuse, FuseFit
from .pipeline import FlowForgePipeline, RunOutcome, build_baselines, build_candidates

__all__ = [
    "FlowFuse",
    "FuseFit",
    "FlowForgePipeline",
    "RunOutcome",
    "build_baselines",
    "build_candidates",
]
