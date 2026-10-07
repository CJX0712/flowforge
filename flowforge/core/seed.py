"""全局确定性：唯一 seed 入口。

踩坑库 §A：多 seed 门槛在 CI 上翻转的常见根因是 seed 未一次设齐；
这里把 numpy legacy 全局 + stdlib random + PYTHONHASHSEED 一次设齐。
"""

from __future__ import annotations

import os
import random

import numpy as np

_SEED: int | None = None


def set_all(seed: int) -> int:
    """设置全局确定性种子，返回 seed 本身。"""
    global _SEED  # noqa: PLW0603  # 模块级单例状态，刻意如此
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed 必须是 int，收到 {type(seed).__name__}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _SEED = int(seed)
    return _SEED


def current() -> int | None:
    return _SEED


def rng(seed: int | None = None) -> np.random.Generator:
    """取得独立 Generator。显式 seed 优先，否则用全局 seed。"""
    if seed is None:
        if _SEED is None:
            raise RuntimeError("未设置全局 seed，请先调用 core.seed.set_all(seed)")
        seed = _SEED
    return np.random.default_rng(int(seed))


__all__ = ["current", "rng", "set_all"]
