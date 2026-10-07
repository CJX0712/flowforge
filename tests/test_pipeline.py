"""管线 / CLI / 配置 / 离线兜底 测试。"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.config import Config
from flowforge.core.errors import ConfigError, EvalError, NegativeKLError, UnknownMethodError
from flowforge.core.seed import set_all
from flowforge.eval.metrics import assert_kl_nonneg, beats, kl_gap, mmd_rbf, summarize, w2_distance
from flowforge.flows.models import GaussianModel
from flowforge.pipeline.benchmark import run_benchmark
from flowforge.pipeline.methods import FUSION_BLOCKS, METHODS, candidate_specs, select_spec


# ------------------------------------------------------------------ 配置
def test_config_rejects_bad_values():
    with pytest.raises(ConfigError):
        Config(iters=0).validate()
    with pytest.raises(ConfigError):
        Config(lr=-1).validate()
    with pytest.raises(ConfigError):
        Config(batch=10**6).validate()
    with pytest.raises(ConfigError):
        Config(n_bins=1).validate()
    with pytest.raises(ConfigError):
        Config(n_components=0).validate()


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("FLOWFORGE_ITERS", "77")
    monkeypatch.setenv("FLOWFORGE_LR", "0.002")
    cfg = Config.from_env()
    assert cfg.iters == 77 and cfg.lr == pytest.approx(0.002)


def test_config_env_bad_value_raises(monkeypatch):
    monkeypatch.setenv("FLOWFORGE_ITERS", "not-a-number")
    with pytest.raises(ConfigError):
        Config.from_env()


# ------------------------------------------------------------------ 指标
def test_kl_gap_shape_mismatch_raises():
    with pytest.raises(EvalError):
        kl_gap(np.zeros(3), np.zeros(4))


def test_kl_gap_nonfinite_raises():
    with pytest.raises(EvalError):
        kl_gap(np.array([np.inf]), np.array([0.0]))


def test_assert_kl_nonneg_raises_on_negative():
    with pytest.raises(NegativeKLError):
        assert_kl_nonneg(-0.5, tol=1e-6, ctx="unit-test")


def test_beats_criterion():
    # 均值差 0.5 > 0.5*(0.2+0.2)=0.2 -> 胜
    assert beats(1.0, 0.2, 1.6, 0.2)
    # 均值差 0.1 <= 0.5*(0.2+0.2)=0.2 -> 不胜
    assert not beats(1.0, 0.2, 1.1, 0.2)


def test_summarize_single_value_std_zero():
    mu, sd = summarize([3.0])
    assert mu == 3.0 and sd == 0.0


def test_w2_and_mmd_zero_for_identical_sets():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(120, 2))
    assert w2_distance(X, X) < 1e-3
    assert mmd_rbf(X, X) < 1e-8


def test_w2_positive_for_shifted_sets():
    rng = np.random.default_rng(10)
    A = rng.normal(size=(120, 2))
    B = rng.normal(size=(120, 2)) + 3.0
    assert w2_distance(A, B) > 0.5


# ------------------------------------------------------------------ 方法注册
def test_methods_registry_has_four_methods():
    assert METHODS == ("gaussian", "affine", "maf", "fusion")
    assert "spline" not in FUSION_BLOCKS


def test_candidate_specs_equal_budget():
    """等参数预算：fusion 的总块数与基线一致。"""
    cfg = Config()
    for m in ("affine", "maf"):
        for s in candidate_specs(m, cfg):
            assert s.n_blocks == cfg.n_blocks
    for s in candidate_specs("fusion", cfg):
        assert s.n_blocks == cfg.n_blocks
        assert s.n_components >= 1


def test_unknown_method_raises():
    with pytest.raises(UnknownMethodError):
        candidate_specs("nope", Config())


def test_select_spec_returns_valid_spec():
    set_all(11)
    cfg = Config(hidden=8, n_blocks=4, iters=40)
    from flowforge.data.densities import build_density

    X = build_density("gmm5").sample(300, np.random.default_rng(11))
    spec = select_spec("fusion", X, cfg, hpo_iters=30)
    assert spec.family == "fusion"
    assert spec.n_components in (1, 2, 4)


# ------------------------------------------------------------------ 端到端
def test_benchmark_smoke_offline():
    """离线兜底：只用 numpy/scipy/sklearn，无网络、无权重下载。"""
    set_all(12)
    cfg = Config(
        n_train=300, n_test=300, iters=40, batch=128, hidden=8, n_blocks=4, n_w2_samples=80
    )
    rows = run_benchmark(cfg, datasets=("gmm5",), methods=("gaussian", "affine"), seeds=(7,))
    assert len(rows) == 2
    for r in rows:
        assert np.isfinite(r.kl)
        assert np.isfinite(r.w2)
        assert r.kl > -1e-6, f"{r.method} 的 KL 为负"


def test_gaussian_baseline_kl_worse_than_flow():
    """高斯必须明显弱于流（否则基准无区分度）。"""
    set_all(13)
    from flowforge.data.densities import build_density

    d = build_density("ring8")
    X = d.sample(3000, np.random.default_rng(13))
    g = GaussianModel().fit(X)
    kl = kl_gap(g.log_prob(X), d.log_prob(X))
    assert kl > 0.1, "高斯基线在多模态数据上应显著更差"


def test_cli_check_command():
    from flowforge.cli import main

    rc = main(["--check"])
    assert rc == 0


def test_cli_quick_runs():
    from flowforge.cli import main

    assert main(["--quick", "--datasets", "gmm5", "--methods", "affine"]) == 0
