"""硬不变量：数学正确性的断言集合（CI 门禁 + demo 自检）。

每条不变量都对应一类"不崩不报错、只静默污染结论"的缺陷。
  I1  往返可逆性    inverse(forward(X)) == X             （机器精度）
  I2  logdet 正确性 解析 log|det| == 数值雅可比 log|det|   （有限差分）
  I3  梯度正确性    解析梯度 == 中心差分                   （有限差分）
  I4  密度归一化    ∫ p_model(x) dx == 1                  （数值积分）
  I5  KL 非负       KL(p_true||p_model) >= 0              （Gibbs）
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.errors import GradientCheckError, LogDetMismatchError
from .metrics import kl_gap


def roundtrip_error(model: Any, X: np.ndarray) -> float:
    """I1：往返可逆性最大绝对误差。"""
    Z, _ = model.forward(X)
    return float(np.max(np.abs(model.inverse(Z) - X)))


def logdet_error(model: Any, X: np.ndarray, eps: float = 1e-6, max_points: int = 8) -> float:
    """I2：解析 log|det| 与数值雅可比行列式的最大偏差。"""
    d = X.shape[1]
    worst = 0.0
    for i in range(min(max_points, X.shape[0])):
        x0 = X[i]
        J = np.zeros((d, d))
        for k in range(d):
            step = np.zeros(d)
            step[k] = eps
            zp, _ = model.forward((x0 + step)[None, :])
            zm, _ = model.forward((x0 - step)[None, :])
            J[:, k] = (zp[0] - zm[0]) / (2 * eps)
        det = np.linalg.det(J)
        if det <= 0:
            raise LogDetMismatchError("雅可比行列式非正，变换不可逆", det=float(det))
        _, ld = model.forward(x0[None, :])
        worst = max(worst, abs(float(np.log(det)) - float(ld[0])))
    return worst


def gradient_error(model: Any, X: np.ndarray, eps: float = 1e-6, max_params: int = 6) -> float:
    """I3：解析梯度 vs 中心差分的最大相对误差。"""
    _f0, grads = model.grads(X)
    params = model.params()
    if len(grads) != len(params):
        raise GradientCheckError("梯度与参数数量不匹配", n_grads=len(grads), n_params=len(params))
    worst = 0.0
    for p, g in zip(params, grads, strict=True):
        fp = p.reshape(-1)
        fg = np.asarray(g).reshape(-1)
        if fp.size != fg.size:
            raise GradientCheckError("梯度与参数形状不匹配", param=fp.size, grad=fg.size)
        for j in range(min(max_params, fp.size)):
            old = fp[j]
            fp[j] = old + eps
            f_plus = model.grads(X)[0]
            fp[j] = old - eps
            f_minus = model.grads(X)[0]
            fp[j] = old
            num = (f_plus - f_minus) / (2 * eps)
            ana = float(fg[j])
            worst = max(worst, abs(num - ana) / max(1.0, abs(num) + abs(ana)))
    return worst


def density_normalization_error(
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    dim: int = 2,
    lo: float = -12.0,
    hi: float = 12.0,
    n: int = 900,
    chunk: int = 20000,
) -> float:
    """I4：数值积分 ∫ p(x) dx 与 1 的绝对偏差（网格积分，**分块**）。

    必须分块：n=900 的二维网格有 810,000 个点，一次性送进 MLP 会分配约
    198MiB（float64 的 (810000, hidden) 中间量），内存紧张时直接
    `ArrayMemoryError`。实测在**干净环境**首次跑 demo 时崩在这里，
    而本机因内存充裕从未暴露 —— 干净环境复现验证的价值所在。
    """
    if dim != 2:
        raise ValueError("当前实现只支持 2 维网格积分")
    g = np.linspace(lo, hi, n)
    cell = (g[1] - g[0]) ** dim
    rows_per = max(1, chunk // n)
    total = 0.0
    for r0 in range(0, n, rows_per):
        mesh = np.meshgrid(g[r0 : r0 + rows_per], g, indexing="ij")
        X = np.stack([mesh[0].ravel(), mesh[1].ravel()], axis=1)
        total += float(np.sum(np.exp(log_prob_fn(X))))
        del mesh, X
    return abs(total * cell - 1.0)


def check_all(
    model: Any,
    X: np.ndarray,
    log_p_true: np.ndarray,
    tol: dict | None = None,
    X_kl: np.ndarray | None = None,
    log_p_true_kl: np.ndarray | None = None,
) -> list[tuple[str, float, bool]]:
    """一次性跑完 I1~I5，返回 [(名称, 实测值, 是否通过)]。

    I5（KL 非负）必须用**大样本**：KL 是蒙特卡洛估计，样本量 24 时标准误可达
    0.1 量级，会把「KL≈0 的好模型」误判为负（实测踩过）。故支持单独的 X_kl。
    """
    tol = tol or {}
    t_rt = tol.get("roundtrip", 1e-8)
    t_ld = tol.get("logdet", 1e-5)
    t_gc = tol.get("gradient", 1e-5)
    t_nm = tol.get("normalize", 5e-2)
    t_kl = tol.get("kl", 1e-6)

    Xk = X if X_kl is None else X_kl
    lk = log_p_true if log_p_true_kl is None else log_p_true_kl

    # 混合流不是单一微分同胚：I1/I2 下放到每个分量（每个分量都是严格的流）
    parts = list(getattr(model, "comps", [model]))

    rt = max(roundtrip_error(p, X) for p in parts)
    ld = max(logdet_error(p, X) for p in parts)
    gc = gradient_error(model, X)
    nz = density_normalization_error(model.log_prob)
    kl = kl_gap(model.log_prob(Xk), lk)
    return [
        ("I1_roundtrip", rt, rt <= t_rt),
        ("I2_logdet", ld, ld <= t_ld),
        ("I3_gradient", gc, gc <= t_gc),
        ("I4_normalize", nz, nz <= t_nm),
        ("I5_kl_nonneg", kl, kl >= -t_kl),
    ]


__all__ = [
    "check_all",
    "density_normalization_error",
    "gradient_error",
    "logdet_error",
    "roundtrip_error",
]
