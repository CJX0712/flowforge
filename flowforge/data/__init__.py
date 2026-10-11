"""Data package: leak-free synthetic density benchmarks.

Author: 晨星
"""

from __future__ import annotations

from .synthetic import REGISTRY, make_dataset, list_datasets

__all__ = ["make_dataset", "list_datasets", "REGISTRY"]
