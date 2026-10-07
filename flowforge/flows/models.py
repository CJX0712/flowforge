"""归一化流模型：组合流 Flow / 混合流 MixtureFlow / 高斯基线 GaussianModel。

统一训练接口（供 training 使用）：
  model.params()          -> List[np.ndarray]，稳定顺序
  model.grads(X, w=None)  -> (f_mean: float, grads: List[np.ndarray] 与 params() 对齐)

预处理标准化只在 **train 折** fit（防泄漏），变换的 log|det| = -sum(log std) 计入 log_prob。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.errors import FlowError
from .bijectors import AffineCoupling, LinearSplineCoupling, MaskedAffineAutoregressive

_LOG2PI = math.log(2.0 * math.pi)


def _grad_list(block: Any, gb: dict[str, np.ndarray]) -> list[np.ndarray]:
    """把 backward 返回的 grads dict 按 params() 顺序对齐。"""
    if isinstance(block, (AffineCoupling, LinearSplineCoupling)):
        return [gb["W1"], gb["b1"], gb["W2"], gb["b2"]]
    if isinstance(block, MaskedAffineAutoregressive):
        out = [gb["s0"], gb["t0"]]
        for i in range(block.dim - 1):
            out += [gb[f"net{i}.W1"], gb[f"net{i}.b1"], gb[f"net{i}.W2"], gb[f"net{i}.b2"]]
        return out
    raise FlowError("未知双射器类型", typ=type(block).__name__)


class Flow:
    """B 个双射器的复合 + 标准正态 base（可选输入标准化）。"""

    def __init__(
        self,
        blocks: Sequence[Any],
        dim: int,
        pre_mean: np.ndarray | None = None,
        pre_std: np.ndarray | None = None,
    ) -> None:
        self.blocks = list(blocks)
        self.dim = int(dim)
        self.pre_mean = np.zeros(dim) if pre_mean is None else np.asarray(pre_mean, dtype=float)
        self.pre_std = np.ones(dim) if pre_std is None else np.asarray(pre_std, dtype=float)
        if np.any(self.pre_std <= 0):
            raise FlowError("标准化 std 必须为正", std=self.pre_std.tolist())

    # ---- 参数 ----
    def nparams(self) -> int:
        return sum(b.nparams() for b in self.blocks)

    def params(self) -> list[np.ndarray]:
        ps: list[np.ndarray] = []
        for b in self.blocks:
            ps += b.params()
        return ps

    def set_params(self, values: Sequence[np.ndarray]) -> None:
        i = 0
        for b in self.blocks:
            for p in b.params():
                p[...] = values[i]
                i += 1

    # ---- 变换 ----
    def _pre(self, X: np.ndarray) -> np.ndarray:
        return (X - self.pre_mean) / self.pre_std

    def _post(self, Xs: np.ndarray) -> np.ndarray:
        return Xs * self.pre_std + self.pre_mean

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = self._pre(X)
        # 标准化的 log|det| = -sum(log std)（逐样本相同）
        ld0 = np.full(X.shape[0], -float(np.sum(np.log(self.pre_std))))
        ld = ld0
        for b in self.blocks:
            Z, lstep = b.forward(Z)
            ld = ld + lstep
        return Z, ld

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        X = Z
        for b in reversed(self.blocks):
            X = b.inverse(X)
        return self._post(X)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        Z, ld = self.forward(X)
        return -0.5 * (self.dim * _LOG2PI + np.sum(Z**2, axis=1)) + ld

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        Z = rng.standard_normal((n, self.dim))
        return self.inverse(Z)

    # ---- 梯度 ----
    def nll_grad(
        self, X: np.ndarray, w: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """返回 (每样本 f=-log p, 每 block 的 grads dict)。梯度按 batch 求和。"""
        if w is None:
            w = np.ones(X.shape[0])
        Z, ld = self.forward(X)
        f = -(-0.5 * (self.dim * _LOG2PI + np.sum(Z**2, axis=1)) + ld)
        g = Z * w[:, None]
        grads: list[dict[str, np.ndarray]] = []
        for b in reversed(self.blocks):
            g, gb = b.backward(g, w)
            grads.append(gb)
        return f, list(reversed(grads))

    def grads(self, X: np.ndarray, w: np.ndarray | None = None) -> tuple[float, list[np.ndarray]]:
        n = X.shape[0]
        f, gblocks = self.nll_grad(X, w)
        out: list[np.ndarray] = []
        for b, gb in zip(self.blocks, gblocks, strict=True):
            out += [g / n for g in _grad_list(b, gb)]
        return float(np.mean(f)), out


class MixtureFlow:
    """独立分量的混合归一化流：log p(x) = logsumexp_k [log pi_k + log p_k(x)]。"""

    def __init__(self, components: Sequence[Flow], logits: np.ndarray) -> None:
        if len(components) == 0:
            raise FlowError("混合流至少需要 1 个分量", n=len(components))
        self.comps = list(components)
        self.logits = np.asarray(logits, dtype=float)
        self.dim = self.comps[0].dim

    # ---- 参数 ----
    def nparams(self) -> int:
        return sum(c.nparams() for c in self.comps) + self.logits.size

    def params(self) -> list[np.ndarray]:
        ps: list[np.ndarray] = []
        for c in self.comps:
            ps += c.params()
        return [*ps, self.logits]

    def set_params(self, values: Sequence[np.ndarray]) -> None:
        i = 0
        for c in self.comps:
            for p in c.params():
                p[...] = values[i]
                i += 1
        self.logits[...] = values[i]

    def _log_weights(self) -> np.ndarray:
        m = float(np.max(self.logits))
        return self.logits - (m + math.log(float(np.sum(np.exp(self.logits - m)))))

    def responsibilities(self, X: np.ndarray) -> np.ndarray:
        lpk = np.stack([c.log_prob(X) for c in self.comps], axis=1)
        lw = self._log_weights()
        a = lw[None, :] + lpk
        m = np.max(a, axis=1, keepdims=True)
        return np.exp(a - m) / np.sum(np.exp(a - m), axis=1, keepdims=True)

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        lpk = np.stack([c.log_prob(X) for c in self.comps], axis=1)
        lw = self._log_weights()
        a = lw[None, :] + lpk
        m = np.max(a, axis=1, keepdims=True)
        return (m + np.log(np.sum(np.exp(a - m), axis=1, keepdims=True))).ravel()

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        pi = np.exp(self._log_weights())
        k = rng.choice(len(self.comps), size=n, p=pi)
        out = np.empty((n, self.dim))
        for j, c in enumerate(self.comps):
            mask = k == j
            if np.any(mask):
                out[mask] = c.sample(int(mask.sum()), rng)
        return out

    def grads(self, X: np.ndarray, w: np.ndarray | None = None) -> tuple[float, list[np.ndarray]]:
        n = X.shape[0]
        lpk = np.stack([c.log_prob(X) for c in self.comps], axis=1)
        lw = self._log_weights()
        a = lw[None, :] + lpk
        m = np.max(a, axis=1, keepdims=True)
        lse = (m + np.log(np.sum(np.exp(a - m), axis=1, keepdims=True))).ravel()
        r = np.exp(a - lse[:, None])  # (n, K) 责任度
        f = -lse
        out: list[np.ndarray] = []
        for k, c in enumerate(self.comps):
            _, gl = c.grads(X, r[:, k])
            out += gl
        pi = np.exp(lw)
        dl = -(r.sum(axis=0) / n)  # d f/d log pi_k
        # log pi = logits - lse(logits)；d logpi_j/d logit_i = delta_ij - pi_i
        out.append(dl - pi * float(np.sum(dl)))
        return float(np.mean(f)), out


class GaussianModel:
    """全协方差高斯 MLE 基线（无流）。解析解，无训练。"""

    name = "gaussian"

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.dim = 0

    def nparams(self) -> int:
        if self.mean is None:
            return 0
        return self.mean.size + self.cov.size

    def fit(self, X: np.ndarray) -> GaussianModel:
        self.dim = X.shape[1]
        self.mean = X.mean(axis=0)
        C = np.cov(X, rowvar=False)
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-6, None)  # 保证正定
        self.cov = (V * w) @ V.T
        self._L = np.linalg.cholesky(self.cov)
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._L))))
        return self

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise FlowError("GaussianModel 尚未 fit")
        dev = np.linalg.solve(self._L, (X - self.mean).T)
        maha = np.sum(dev**2, axis=0)
        return -0.5 * (self.dim * _LOG2PI + self._logdet + maha)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.mean is None:
            raise FlowError("GaussianModel 尚未 fit")
        return self.mean + rng.standard_normal((n, self.dim)) @ self._L.T


__all__ = ["Flow", "GaussianModel", "MixtureFlow", "_grad_list"]
