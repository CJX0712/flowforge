"""接口契约（Protocol）：调用单向无环 cli → pipeline → {data,flows,training,eval} → core。

Bijector 契约（硬）：
  forward(X) -> (Z, logdet)      数据域 → 隐空间，logdet = log|det dZ/dX|
  inverse(Z) -> X                必须满足 inverse(forward(X)) == X（机器精度）
  backward(gZ, w) -> (gX, grads) 解析梯度；w 为每样本责任度权重
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Bijector(Protocol):
    """可逆变换块。"""

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def inverse(self, Z: np.ndarray) -> np.ndarray: ...

    def backward(
        self, gZ: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]: ...

    def nparams(self) -> int: ...

    def params(self) -> list: ...

    def set_params(self, values: dict[str, np.ndarray]) -> None: ...


@runtime_checkable
class Density(Protocol):
    """可解析求对数密度的目标分布。"""

    @property
    def name(self) -> str: ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def log_prob(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class GenerativeModel(Protocol):
    """归一化流模型（单流或混合流）。"""

    def log_prob(self, X: np.ndarray) -> np.ndarray: ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def nparams(self) -> int: ...

    def nll_grad(self, X: np.ndarray) -> tuple[np.ndarray, Any]: ...


@runtime_checkable
class Method(Protocol):
    """可参与 benchmark 的方法。"""

    @property
    def name(self) -> str: ...

    def fit(self, X: np.ndarray, cfg: Any, seed: int) -> Method: ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """返回 log p_model(x)。"""
        ...


__all__ = ["Bijector", "Density", "GenerativeModel", "Method"]
