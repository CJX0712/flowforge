"""硬不变量测试：数学正确性的回归防线。

每条测试对应一类「不崩不报错、只静默污染结论」的缺陷。
"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.seed import set_all
from flowforge.data.densities import DATASETS, build_density
from flowforge.eval.invariants import (
    check_all,
    density_normalization_error,
    gradient_error,
    logdet_error,
    roundtrip_error,
)
from flowforge.flows import (
    AffineCoupling,
    Flow,
    GaussianModel,
    LinearSplineCoupling,
    MaskedAffineAutoregressive,
    MixtureFlow,
)

DIM = 2
MASKS = [np.array([True, False]), np.array([False, True])]


def _flow(kind: str, seed: int = 0, n_blocks: int = 2, hidden: int = 8, n_bins: int = 4) -> Flow:
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(n_blocks):
        m = MASKS[i % DIM]
        if kind == "affine":
            blocks.append(AffineCoupling(m, hidden, rng))
        elif kind == "maf":
            blocks.append(MaskedAffineAutoregressive(DIM, hidden, rng, reverse=(i % 2 == 1)))
        elif kind == "spline":
            blocks.append(LinearSplineCoupling(m, hidden, n_bins, rng))
    return Flow(blocks, DIM)


@pytest.mark.parametrize("kind", ["affine", "maf", "spline"])
def test_i1_roundtrip(kind):
    set_all(1)
    m = _flow(kind)
    X = np.random.default_rng(1).normal(size=(16, DIM))
    assert roundtrip_error(m, X) < 1e-9


@pytest.mark.parametrize("kind", ["affine", "maf", "spline"])
def test_i2_logdet_matches_numerical_jacobian(kind):
    set_all(2)
    m = _flow(kind)
    X = np.random.default_rng(2).normal(size=(4, DIM))
    assert logdet_error(m, X) < 1e-5


@pytest.mark.parametrize("kind", ["affine", "maf", "spline"])
def test_i3_gradient_matches_central_difference(kind):
    set_all(3)
    m = _flow(kind)
    X = np.random.default_rng(3).normal(size=(12, DIM))
    assert gradient_error(m, X, max_params=8) < 1e-5


def test_i3_gradient_mixture():
    set_all(4)
    comps = [_flow("affine", seed=4 + j, n_blocks=2, hidden=8) for j in range(3)]
    mm = MixtureFlow(comps, np.zeros(3))
    X = np.random.default_rng(4).normal(size=(10, DIM))
    assert gradient_error(mm, X, max_params=6) < 1e-5


def test_i4_density_normalizes():
    set_all(5)
    m = _flow("affine", seed=5, n_blocks=3, hidden=8)
    assert density_normalization_error(m.log_prob, n=400) < 5e-2


def test_i4_mixture_normalizes():
    set_all(6)
    comps = [_flow("affine", seed=6 + j, n_blocks=2, hidden=8) for j in range(3)]
    mm = MixtureFlow(comps, np.zeros(3))
    assert density_normalization_error(mm.log_prob, n=400) < 5e-2


def test_i5_kl_nonnegative_on_all_datasets():
    """Gibbs 不等式：KL(p_true||p_model) >= 0。负值=真值或模型算错。

    两个必须遵守的细节（否则测试自己就会假失败）：
    1) 模型必须在**独立**样本上 fit —— 同一份数据既 fit 又评估 KL 会产生
       in-sample 乐观偏差，使估计量可能为负（踩坑库 §F 记录的经典陷阱）。
    2) KL 是蒙特卡洛估计，**期望**非负但有限样本估计可略负；
       容差按 5 倍标准误给出，而不是硬取 0。
    """
    set_all(7)
    for name in DATASETS:
        dens = build_density(name)
        X_fit = dens.sample(2000, np.random.default_rng(7))
        X_eval = dens.sample(2000, np.random.default_rng(707))
        m = GaussianModel().fit(X_fit)
        diffs = dens.log_prob(X_eval) - m.log_prob(X_eval)
        kl = float(np.mean(diffs))
        tol = 5.0 * float(np.std(diffs, ddof=1)) / np.sqrt(X_eval.shape[0])
        assert kl >= -tol, f"{name}: KL={kl} < -{tol}"


def test_check_all_runs_and_reports_five_items():
    set_all(8)
    dens = build_density("banana")
    X = dens.sample(300, np.random.default_rng(8))
    m = _flow("affine", seed=8, n_blocks=2, hidden=8)
    res = check_all(m, X[:16], dens.log_prob(X[:16]))
    assert len(res) == 5
    names = [n for n, _, _ in res]
    assert names == ["I1_roundtrip", "I2_logdet", "I3_gradient", "I4_normalize", "I5_kl_nonneg"]


def test_gaussian_model_matches_analytic_density():
    """高斯基线必须与 scipy 解析密度一致（交叉验证参照实现）。"""
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(9)
    mu = np.array([1.0, -2.0])
    C = np.array([[2.0, 0.4], [0.4, 1.0]])
    X = rng.multivariate_normal(mu, C, size=4000)
    m = GaussianModel().fit(X)
    ref = multivariate_normal.logpdf(X, mean=m.mean, cov=m.cov)
    assert np.max(np.abs(m.log_prob(X) - ref)) < 1e-8


def test_maf_alternating_order_changes_conditioning():
    """交替顺序必须改变变换（否则说明 reverse 参数没生效——结构性表达力缺陷）。"""
    set_all(10)
    X = np.random.default_rng(10).normal(size=(8, DIM))
    a = MaskedAffineAutoregressive(DIM, 8, np.random.default_rng(10), reverse=False)
    b = MaskedAffineAutoregressive(DIM, 8, np.random.default_rng(10), reverse=True)
    b.s0[...] = a.s0
    b.t0[...] = a.t0
    for na, nb_ in zip(a.nets, b.nets, strict=True):
        nb_.W1[...] = na.W1
        nb_.b1[...] = na.b1
        nb_.W2[...] = na.W2
        nb_.b2[...] = na.b2
    za, _ = a.forward(X)
    zb, _ = b.forward(X)
    # 反序后第 0/1 维的角色互换，结果必须不同
    assert not np.allclose(za, zb)


def test_spline_min_bin_width_prevents_degenerate_slope():
    """min_bin_width 必须保证每段宽度有下界（否则 slope -> inf 致训练发散）。"""
    rng = np.random.default_rng(11)
    b = LinearSplineCoupling(MASKS[0], 8, 4, rng, bound=4.0, min_bin_width=0.05)
    X = rng.normal(size=(32, DIM)) * 2.0
    b.forward(X)
    widths = b._c["widths"]
    assert float(widths.min()) >= 0.05 - 1e-12
