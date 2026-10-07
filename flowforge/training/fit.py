"""模型构建与拟合。

标准化（防泄漏）：pre_mean / pre_std **只在 train 折** 统计，
其 log|det| = -sum(log std) 已计入 Flow.log_prob。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.errors import ConfigError
from ..core.types import FlowSpec
from ..flows.bijectors import AffineCoupling, LinearSplineCoupling, MaskedAffineAutoregressive
from ..flows.models import Flow, GaussianModel, MixtureFlow
from .optim import Adam, lr_schedule

BLOCK_TYPES = ("affine", "maf", "spline")


def _make_block(
    block_type: str, dim: int, hidden: int, index: int, n_bins: int, rng: np.random.Generator
) -> Any:
    """按类型与序号造一个双射器（mask 按 index 交替）。"""
    if block_type == "maf":
        # 块间交替自回归顺序：否则首维永远无法以其它维为条件（表达力结构性缺失）
        return MaskedAffineAutoregressive(dim, hidden, rng, reverse=(index % 2 == 1))
    mask = np.zeros(dim, dtype=bool)
    mask[index % dim] = True
    if block_type == "affine":
        return AffineCoupling(mask, hidden, rng)
    if block_type == "spline":
        return LinearSplineCoupling(mask, hidden, n_bins, rng)
    raise ConfigError("未知块类型", block_type=block_type, available=list(BLOCK_TYPES))


def make_blocks(
    block_type: str, dim: int, hidden: int, n_blocks: int, n_bins: int, rng: np.random.Generator
) -> list[Any]:
    if n_blocks <= 0:
        raise ConfigError("n_blocks 必须 > 0", n_blocks=n_blocks)
    return [_make_block(block_type, dim, hidden, i, n_bins, rng) for i in range(n_blocks)]


def standardize_stats(X: np.ndarray) -> tuple:
    """train 折统计量（防泄漏）。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def randomize_params(
    params: Sequence[np.ndarray], rng: np.random.Generator, scale: float = 1.0
) -> None:
    """原地重初始化（用于死组件复活）。"""
    for p in params:
        if p.ndim == 2:
            s = scale * np.sqrt(2.0 / (p.shape[0] + p.shape[1]))
            p[...] = rng.normal(0.0, s, p.shape)
        else:
            p[...] = 0.0


def build_flow(
    spec: FlowSpec,
    dim: int,
    X_train: np.ndarray,
    rng: np.random.Generator,
    n_blocks: int | None = None,
) -> Flow:
    """按 spec 构造单流。n_blocks 可覆盖（用于混合流分摊预算）。"""
    nb = spec.n_blocks if n_blocks is None else int(n_blocks)
    blocks = make_blocks(spec.block_type, dim, spec.hidden, nb, spec.n_bins, rng)
    mean, std = standardize_stats(X_train)
    return Flow(blocks, dim, pre_mean=mean, pre_std=std)


def build_mixture(
    spec: FlowSpec, dim: int, X_train: np.ndarray, rng: np.random.Generator
) -> MixtureFlow:
    """混合流：把 n_blocks 总预算平均分给 n_components 个分量（等参数对照）。"""
    K = int(spec.n_components)
    if K < 1:
        raise ConfigError("n_components 必须 >= 1", n_components=K)
    per = max(1, int(spec.n_blocks) // K)
    comps = [build_flow(spec, dim, X_train, rng, n_blocks=per) for _ in range(K)]
    return MixtureFlow(comps, np.zeros(K))


def build_model(spec: FlowSpec, dim: int, X_train: np.ndarray, rng: np.random.Generator):
    """按 family 构造模型。gaussian 走单独路径。"""
    if spec.family == "fusion":
        if spec.n_components <= 1:
            return build_flow(spec, dim, X_train, rng)
        return build_mixture(spec, dim, X_train, rng)
    if spec.family in ("affine", "maf", "spline"):
        s2 = FlowSpec(
            family=spec.family,
            block_type=spec.family,
            n_blocks=spec.n_blocks,
            hidden=spec.hidden,
            n_bins=spec.n_bins,
            seed=spec.seed,
        )
        return build_flow(s2, dim, X_train, rng)
    raise ConfigError("未知 family", family=spec.family)


def fit(
    model,
    X: np.ndarray,
    seed: int = 7,
    iters: int = 600,
    lr: float = 5e-3,
    lr_end: float = 3e-4,
    batch: int = 512,
    clip: float = 5.0,
    revive: bool = False,
    dead_eps: float = 0.02,
    revive_every: int = 100,
) -> dict[str, Any]:
    """训练任意实现了 params()/grads() 的模型。返回诊断信息。"""
    if X.ndim != 2:
        raise ConfigError("X 必须是 (n, d)", shape=X.shape)
    n = X.shape[0]
    bs = int(min(batch, n))
    opt = Adam(model.params(), lr=lr)
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    last = float("nan")
    revived = 0
    for it in range(int(iters)):
        cur = lr_schedule(it, int(iters), lr, lr_end)
        Xb = X[rng.permutation(n)[:bs]]
        f, grads = model.grads(Xb)
        last = float(f)
        if revive and isinstance(model, MixtureFlow) and it > 0 and it % revive_every == 0:
            r = model.responsibilities(Xb)
            mean_r = r.mean(axis=0)
            dead = np.where(mean_r < dead_eps)[0]
            for k in dead:
                randomize_params(model.comps[k].params(), rng, scale=1.0)
                model.logits[k] = 0.0
                revived += 1
        opt.step(grads, lr=cur, clip=clip)
    return {
        "final_nll": last,
        "train_sec": time.perf_counter() - t0,
        "revived_components": revived,
        "iters": int(iters),
    }


__all__ = [
    "BLOCK_TYPES",
    "GaussianModel",
    "build_flow",
    "build_mixture",
    "build_model",
    "fit",
    "make_blocks",
    "randomize_params",
    "standardize_stats",
]
