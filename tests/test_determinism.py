"""确定性测试：同 seed 两次运行必须逐位一致。"""

from __future__ import annotations

import numpy as np

from flowforge.core.config import Config
from flowforge.core.seed import set_all
from flowforge.pipeline.benchmark import run_benchmark


def _core_metrics(rows):
    """剔除耗时等非确定性字段，只比核心指标。"""
    out = {}
    for r in rows:
        out[f"{r.dataset}|{r.method}|{r.seed}"] = (
            round(r.kl, 12),
            round(r.w2, 10),
            round(r.mmd, 12),
            r.n_params,
        )
    return out


def test_benchmark_is_bit_identical_across_runs():
    set_all(7)
    cfg = Config(
        n_train=400, n_test=400, iters=60, batch=128, hidden=8, n_blocks=4, n_w2_samples=100
    )
    kw = dict(datasets=("gmm5", "banana"), methods=("affine", "fusion"), seeds=(7, 11))
    a = run_benchmark(cfg, **kw)
    b = run_benchmark(cfg, **kw)
    ma, mb = _core_metrics(a), _core_metrics(b)
    assert ma == mb, "同 seed 两次运行的 benchmark 核心指标不一致"


def test_seed_setter_is_idempotent_and_reports():
    assert set_all(123) == 123
    from flowforge.core import seed as seedmod

    assert seedmod.current() == 123
    r1 = seedmod.rng().standard_normal(5)
    set_all(123)
    r2 = seedmod.rng().standard_normal(5)
    assert np.array_equal(r1, r2)


def test_seed_requires_call_before_rng():
    from flowforge.core import seed as seedmod

    seedmod._SEED = None
    try:
        seedmod.rng()
    except RuntimeError:
        return
    raise AssertionError("未设置 seed 时 rng() 应抛 RuntimeError")
