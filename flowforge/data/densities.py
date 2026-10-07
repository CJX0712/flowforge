"""可解析求对数密度的合成 2D 分布（金标准来源）。

设计铁律：每个分布的 sample() 与 log_prob() 必须**严格同源**——
任何一处方差/雅可比写错，KL 金标准立刻给出负值（踩坑库实例：banana 的
z1 方差 1.6 被 log_prob 按单位方差计算，导致 KL 估计系统性偏 −0.065 nats）。
因此每个分布都带 `.normalization_error()` 数值积分自检（∫p dx ≈ 1）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.errors import DataError, ShapeMismatchError, UnknownDatasetError

DIM = 2
_LOG2PI = math.log(2.0 * math.pi)


def _lse_rows(a: np.ndarray) -> np.ndarray:
    """沿 axis=1 做 log-sum-exp，返回 (n,)。"""
    m = np.max(a, axis=1, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=1, keepdims=True))).ravel()


def _mvnormal_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """多元高斯对数密度（Cholesky，数值稳定）。"""
    L = np.linalg.cholesky(cov)
    dev = np.linalg.solve(L, (X - mean).T)  # (d, n)
    maha = np.sum(dev**2, axis=0)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (X.shape[1] * _LOG2PI + logdet + maha)


class _BaseDensity:
    name = "base"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _check(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != DIM:
            raise ShapeMismatchError(
                "输入必须是 (n, 2)", got_shape=tuple(np.asarray(X).shape), want_dim=DIM
            )

    def normalization_error(self, lo: float = -12.0, hi: float = 12.0, n: int = 900) -> float:
        """数值积分 ∫ p(x) dx 与 1 的相对偏差。用于单测断言。"""
        g = np.linspace(lo, hi, n)
        X1, X2 = np.meshgrid(g, g, indexing="ij")
        X = np.stack([X1.ravel(), X2.ravel()], axis=1)
        p = np.exp(self.log_prob(X))
        cell = (g[1] - g[0]) ** 2
        return abs(float(p.sum() * cell) - 1.0)

    def metadata(self) -> dict[str, Any]:
        return {"family": self.name}


class GaussianMixture(_BaseDensity):
    """等权高斯混合。多模态、可解析。"""

    def __init__(
        self,
        centers: np.ndarray,
        cov: float | np.ndarray = 0.25,
        weights: np.ndarray | None = None,
        name: str = "gmm",
    ) -> None:
        self.centers = np.asarray(centers, dtype=float)
        self.name = name
        if self.centers.ndim != 2 or self.centers.shape[1] != DIM:
            raise DataError("centers 必须是 (K, 2)", shape=self.centers.shape)
        if np.isscalar(cov):
            self.cov = float(cov) * np.eye(DIM)
        else:
            self.cov = np.asarray(cov, dtype=float)
        if weights is None:
            self.weights = np.full(len(self.centers), 1.0 / len(self.centers))
        else:
            w = np.asarray(weights, dtype=float)
            if abs(w.sum()) < 1e-12:
                raise DataError("权重和不能为 0")
            self.weights = w / w.sum()
        self.L = np.linalg.cholesky(self.cov)
        self.log_w = np.log(self.weights)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = rng.choice(len(self.weights), size=n, p=self.weights)
        Z = rng.standard_normal((n, DIM))
        return self.centers[k] + Z @ self.L.T

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        self._check(X)
        comps = np.stack([_mvnormal_logpdf(X, mu, self.cov) for mu in self.centers], axis=1)
        return _lse_rows(self.log_w[None, :] + comps)

    def metadata(self) -> dict[str, Any]:
        return {"family": "gmm", "n_components": len(self.centers)}


class Banana(_BaseDensity):
    """x1 = z1, x2 = z2 + b(z1² − var1)，z ~ N(0, diag(var1, 1))。det J = 1。

    注意：z1 的方差是 var1 而非 1 —— log_prob 必须带 -0.5*log(var1) 项。
    """

    name = "banana"

    def __init__(self, b: float = 0.35, var1: float = 1.6) -> None:
        self.b = float(b)
        self.var1 = float(var1)
        if self.var1 <= 0:
            raise DataError("var1 必须 > 0", var1=self.var1)

    def _z(self, X: np.ndarray) -> np.ndarray:
        z1 = X[:, 0]
        z2 = X[:, 1] - self.b * (z1**2 - self.var1)
        return np.stack([z1, z2], axis=1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z1 = rng.standard_normal(n) * math.sqrt(self.var1)
        z2 = rng.standard_normal(n)
        x2 = z2 + self.b * (z1**2 - self.var1)
        return np.stack([z1, x2], axis=1)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        self._check(X)
        Z = self._z(X)
        lp1 = -0.5 * (_LOG2PI + math.log(self.var1) + Z[:, 0] ** 2 / self.var1)
        lp2 = -0.5 * (_LOG2PI + Z[:, 1] ** 2)
        return lp1 + lp2

    def metadata(self) -> dict[str, Any]:
        return {"family": "banana", "b": self.b, "var1": self.var1}


class Funnel(_BaseDensity):
    """Neal (2003) 漏斗：x1 ~ N(0, s²)，x2 ~ N(0, exp(x1))。变尺度重尾。"""

    name = "funnel"

    def __init__(self, s: float = 1.5) -> None:
        self.s = float(s)
        if self.s <= 0:
            raise DataError("s 必须 > 0", s=self.s)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x1 = rng.standard_normal(n) * self.s
        x2 = rng.standard_normal(n) * np.exp(0.5 * x1)
        return np.stack([x1, x2], axis=1)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        self._check(X)
        x1, x2 = X[:, 0], X[:, 1]
        lp1 = -0.5 * (_LOG2PI + 2.0 * math.log(self.s) + (x1 / self.s) ** 2)
        v = np.exp(x1)  # 方差随 x1 指数增长
        lp2 = -0.5 * (_LOG2PI + np.log(v) + x2**2 / v)
        return lp1 + lp2

    def metadata(self) -> dict[str, Any]:
        return {"family": "funnel", "s": self.s}


class Swirl(_BaseDensity):
    """x = R(a‖z‖) z，z ~ N(0, I)。极坐标下 (r, φ) → (r, φ + a·r)，雅可比行列式恒为 1。"""

    name = "swirl"

    def __init__(self, a: float = 2.0) -> None:
        self.a = float(a)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        self._check(X)
        Z = self._z(X)
        return -0.5 * (DIM * _LOG2PI + np.sum(Z**2, axis=1))

    def _z(self, X: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(X, axis=1)
        th = -self.a * r
        c, s = np.cos(th), np.sin(th)
        return np.stack([c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]], axis=1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        Z = rng.standard_normal((n, DIM))
        r = np.linalg.norm(Z, axis=1)
        th = self.a * r
        c, s = np.cos(th), np.sin(th)
        return np.stack([c * Z[:, 0] - s * Z[:, 1], s * Z[:, 0] + c * Z[:, 1]], axis=1)

    def metadata(self) -> dict[str, Any]:
        return {"family": "swirl", "a": self.a}


def _ring_centers(k: int, radius: float) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def build_density(name: str) -> _BaseDensity:
    """按名字构造密度。未知名字抛 UnknownDatasetError。"""
    if name == "gmm5":
        return GaussianMixture(_ring_centers(5, 2.2), cov=0.28, name="gmm5")
    if name == "ring8":
        return GaussianMixture(_ring_centers(8, 2.6), cov=0.16, name="ring8")
    if name == "banana":
        return Banana()
    if name == "funnel":
        return Funnel()
    if name == "swirl":
        return Swirl()
    raise UnknownDatasetError(
        "未知数据集", name=name, available=["gmm5", "ring8", "banana", "funnel", "swirl"]
    )


DATASETS = ("gmm5", "ring8", "banana", "funnel", "swirl")

__all__ = [
    "DATASETS",
    "DIM",
    "Banana",
    "Funnel",
    "GaussianMixture",
    "Swirl",
    "_lse_rows",
    "_mvnormal_logpdf",
    "build_density",
]
