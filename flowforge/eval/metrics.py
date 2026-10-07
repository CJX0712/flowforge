"""评测指标。

金标准（本系统的核心）：合成密度可解析求 log p_true(x)，于是
    KL(p_true || p_model) = E[log p_true - log p_model] = NLL_model - NLL_true >= 0
（Gibbs 不等式）。这是**有下界 0 的绝对标尺**，不需要"更强基线"作参照。
负的 KL 估计 = 真值密度或模型密度算错，本系统已用它抓到过真实 bug。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import EvalError, NegativeKLError


def kl_gap(log_p_model: np.ndarray, log_p_true: np.ndarray) -> float:
    """KL(p_true || p_model) 的蒙特卡洛估计。理论 >= 0。"""
    a = np.asarray(log_p_model, dtype=float)
    b = np.asarray(log_p_true, dtype=float)
    if a.shape != b.shape:
        raise EvalError("两份对数密度形状不一致", model=a.shape, true=b.shape)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise EvalError("对数密度含非有限值")
    return float(np.mean(b - a))


def assert_kl_nonneg(value: float, tol: float = 1e-6, ctx: str = "") -> None:
    """硬不变量：KL 不得为负（超出容差即抛错，绝不静默）。"""
    if not np.isfinite(value):
        raise NegativeKLError("KL 非有限值", value=value, ctx=ctx)
    if value < -tol:
        raise NegativeKLError(
            "KL 为负，违反 Gibbs 不等式：真值密度或模型密度有误", value=value, tol=tol, ctx=ctx
        )


def _cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    d2 = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2.0 * X @ Y.T
    return np.maximum(d2, 0.0)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _ot_entropic(C: np.ndarray, reg: float, iters: int = 80) -> float:
    """熵正则最优传输的对数域 Sinkhorn，返回传输代价。

    iters 默认 80：2D、n<=300 时边残差已收敛到 <1e-4，再多是纯烧 CPU
    （实测 200 iters × 600 样本 会占掉整个 benchmark 七成时间）。
    """
    n, m = C.shape
    la = np.full(n, -np.log(n))
    lb = np.full(m, -np.log(m))
    M = -C / reg
    lu = np.zeros(n)
    lv = np.zeros(m)
    for _ in range(iters):
        lu = la - _logsumexp(M + lv[None, :], axis=1)
        lv = lb - _logsumexp(M.T + lu[:, None], axis=1)
    log_plan = M + lu[:, None] + lv[None, :]
    plan = np.exp(log_plan)
    return float(np.sum(plan * C))


def sinkhorn_divergence(X: np.ndarray, Y: np.ndarray, reg: float | None = None) -> float:
    """去偏 Sinkhorn 散度 S_eps(a,b) = OT(a,b) - 0.5 OT(a,a) - 0.5 OT(b,b)。

    踩坑库 §F：裸 Sinkhorn 有熵偏差，不能直接当 W2 用；去偏后才是合法的分布散度。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Cxy = _cost_matrix(X, Y)
    if reg is None:
        med = float(np.median(Cxy))
        reg = max(med * 0.05, 1e-3)
    Cxx = _cost_matrix(X, X)
    Cyy = _cost_matrix(Y, Y)
    s = _ot_entropic(Cxy, reg) - 0.5 * _ot_entropic(Cxx, reg) - 0.5 * _ot_entropic(Cyy, reg)
    return float(max(s, 0.0))


def w2_distance(X: np.ndarray, Y: np.ndarray, reg: float | None = None) -> float:
    """以去偏 Sinkhorn 散度的平方根作为 W2 估计（平方代价矩阵）。"""
    return float(np.sqrt(sinkhorn_divergence(X, Y, reg)))


def mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """RBF 核 MMD²（无偏估计，带宽取中位数启发式）。"""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.vstack([X, Y])
    C = _cost_matrix(Z, Z)
    med = float(np.median(C))
    sigma2 = med if med > 1e-9 else 1.0
    K = np.exp(-C / (2.0 * sigma2))
    nx, ny = X.shape[0], Y.shape[0]
    kxx = K[:nx, :nx]
    kyy = K[nx:, nx:]
    kxy = K[:nx, nx:]
    # 无偏：去掉对角项
    t1 = (kxx.sum() - np.trace(kxx)) / (nx * (nx - 1)) if nx > 1 else 0.0
    t2 = (kyy.sum() - np.trace(kyy)) / (ny * (ny - 1)) if ny > 1 else 0.0
    t3 = kxy.sum() / (nx * ny)
    return float(max(t1 + t2 - 2.0 * t3, 0.0))


def summarize(values) -> tuple[float, float]:
    """(mean, std)。std 用样本标准差（ddof=1），单值为 0。"""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise EvalError("空序列无法聚合")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(np.mean(v)), float(np.std(v, ddof=1))


def beats(a_mean: float, a_std: float, b_mean: float, b_std: float) -> bool:
    """踩坑库 §A：胜出判据 = 均值差 > 0.5*(sigma1 + sigma2)。"""
    return (b_mean - a_mean) > 0.5 * (a_std + b_std)


__all__ = [
    "assert_kl_nonneg",
    "beats",
    "kl_gap",
    "mmd_rbf",
    "sinkhorn_divergence",
    "summarize",
    "w2_distance",
]
