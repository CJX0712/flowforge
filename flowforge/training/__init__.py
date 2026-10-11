"""Training package: Adam + maximum-likelihood fitting with early stopping.

Author: 晨星
"""

from __future__ import annotations

from .trainer import Adam, FitResult, collect_pointers, fit

__all__ = ["Adam", "FitResult", "collect_pointers", "fit"]
