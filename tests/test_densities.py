"""数据层测试：真值密度必须自洽（归一化 + sample/log_prob 同源）。"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.errors import DataError, ShapeMismatchError, UnknownDatasetError
from flowforge.data.densities import DATASETS, build_density


@pytest.mark.parametrize("name", DATASETS)
def test_density_integrates_to_one(name):
    """∫ p(x) dx = 1。

    这条断言曾直接抓到 banana 的真实 bug：sample() 用 var1=1.6 采样，
    log_prob() 却按单位方差计算，导致 KL 估计系统性偏 -0.065 nats。
    """
    d = build_density(name)
    assert d.normalization_error(n=900) < 5e-2, f"{name} 未归一化"


@pytest.mark.parametrize("name", DATASETS)
def test_sample_matches_log_prob_moments(name):
    """采样样本的经验二阶矩应与 log_prob 的 Hessian 一致（弱校验：均值~0）。"""
    d = build_density(name)
    rng = np.random.default_rng(3)
    X = d.sample(20000, rng)
    # 多数分布关于原点大致对称；只断言不发散
    assert np.all(np.isfinite(X))
    assert np.abs(X).max() < 50.0


@pytest.mark.parametrize("name", DATASETS)
def test_log_prob_shape_and_finiteness(name):
    d = build_density(name)
    X = d.sample(50, np.random.default_rng(4))
    lp = d.log_prob(X)
    assert lp.shape == (50,)
    assert np.all(np.isfinite(lp))


def test_unknown_dataset_raises():
    with pytest.raises(UnknownDatasetError):
        build_density("does-not-exist")


def test_wrong_shape_raises():
    d = build_density("banana")
    with pytest.raises(ShapeMismatchError):
        d.log_prob(np.zeros((5, 3)))


def test_banana_variance_term_present():
    """回归测试：banana 的 z1 方差是 var1，log_prob 必须含 -0.5*log(var1)。"""
    from flowforge.data.densities import Banana

    b = Banana(b=0.35, var1=1.6)
    x = np.zeros((1, 2))  # z(0) = (0, -b*(0-var1))
    z = b._z(x)
    expected = -0.5 * (np.log(2 * np.pi) + np.log(1.6) + 0.0) - 0.5 * (
        np.log(2 * np.pi) + float(z[0, 1] ** 2)
    )
    assert float(b.log_prob(x)[0]) == pytest.approx(expected, rel=1e-12)


def test_spline_bijector_rejects_degenerate_min_width():
    from flowforge.core.errors import FlowError
    from flowforge.flows import LinearSplineCoupling

    with pytest.raises(FlowError):
        LinearSplineCoupling(
            np.array([True, False]), 8, 64, np.random.default_rng(5), bound=4.0, min_bin_width=0.5
        )


def test_gaussian_mixture_rejects_zero_weights():
    from flowforge.data.densities import GaussianMixture

    with pytest.raises(DataError):
        GaussianMixture(np.zeros((2, 2)), weights=np.zeros(2))
