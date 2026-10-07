"""优化器：Adam + 学习率指数衰减 + 全局梯度裁剪。

踩坑库 §F：噪声/不稳定会让「方法看起来无效」。实测归一化流在无衰减时
KL 非单调回弹（banana 400 iters 0.0092 → 800 iters 0.0297）；加入衰减 + 裁剪后单调收敛。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class Adam:
    """标准 Adam，支持每步覆盖 lr 与全局范数裁剪。"""

    def __init__(
        self,
        params: Sequence[np.ndarray],
        lr: float = 3e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr 必须 > 0")
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2, self.eps = float(b1), float(b2), float(eps)
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def step(
        self, grads: Sequence[np.ndarray], lr: float | None = None, clip: float = 0.0
    ) -> float:
        """执行一步。返回裁剪前的梯度全局范数。"""
        lr = self.lr if lr is None else float(lr)
        self.t += 1
        gnorm = math.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in grads))
        if clip > 0.0 and gnorm > clip and gnorm > 0:
            scale = clip / gnorm
            grads = [np.asarray(g) * scale for g in grads]
        for i, (par, g) in enumerate(zip(self.params, grads, strict=True)):
            grad = np.asarray(g)
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (grad * grad)
            mh = self.m[i] / (1.0 - self.b1**self.t)
            vh = self.v[i] / (1.0 - self.b2**self.t)
            par -= lr * mh / (np.sqrt(vh) + self.eps)  # noqa: PLW2901  # 原地更新参数
        return gnorm


def lr_schedule(it: int, total: int, lr0: float, lr1: float) -> float:
    """指数衰减：lr0 -> lr1。"""
    if total <= 1:
        return lr0
    frac = it / (total - 1)
    return lr0 * (lr1 / lr0) ** frac


__all__ = ["Adam", "lr_schedule"]
