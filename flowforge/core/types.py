"""FlowForge 核心数据结构（dataclass，只读优先）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DensitySpec:
    """一个合成密度的规格。"""

    name: str
    n_train: int = 2000
    n_test: int = 2000
    train_seed: int = 7
    test_seed: int = 10007
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """采样 + 解析对数密度。"""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    log_prob: Any  # Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.X_train.shape[1])


@dataclass(frozen=True)
class FlowSpec:
    """方法规格：族 + 结构超参。"""

    family: str  # planar | radial | affine | maf | spline | fusion | gaussian
    n_blocks: int = 8
    hidden: int = 32
    n_components: int = 1  # >1 即混合流
    block_type: str = "affine"
    n_bins: int = 8  # spline 用
    entropy_reg: float = 0.0  # 混合流责任度熵正则（负号=鼓励均匀）
    seed: int = 7


@dataclass
class FitResult:
    """一次拟合的产出。"""

    method: str
    dataset: str
    seed: int
    spec: FlowSpec
    n_params: int
    kl_gap: float  # KL(p_true || p_model) 估计，>=0
    nll: float
    nll_true: float
    w2: float | None = None
    mmd: float | None = None
    train_sec: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodSummary:
    """多 seed 聚合。"""

    method: str
    dataset: str
    kl_mean: float
    kl_std: float
    seeds: list[int]
    n_params: int = 0
    w2_mean: float | None = None
    w2_std: float | None = None
    mmd_mean: float | None = None


__all__ = ["Dataset", "DensitySpec", "FitResult", "FlowSpec", "MethodSummary"]
