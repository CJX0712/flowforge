"""Pipeline-level tests (small configs; the real numbers come from the demo).

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.config import Config
from flowforge.core.errors import ConfigError
from flowforge.core.seed import set_all
from flowforge.pipeline.pipeline import (
    FLAGSHIP,
    FlowForgePipeline,
    build_baselines,
    build_candidates,
)

TINY = Config(seed=1, epochs=12, n_train=300, n_val=120, n_test=200, hidden=24, patience=8)


def test_run_one_returns_all_models():
    set_all(1)
    pip = FlowForgePipeline(TINY)
    from flowforge.data.preprocess import standardize_dataset
    from flowforge.data.synthetic import make_dataset

    ds = make_dataset("torus3", n_train=300, n_val=120, n_test=200, seed=1)
    ds, _ = standardize_dataset(ds)
    out = pip.run_one(ds, seed=1)
    for key in ("RealNVP-b6", "MAF-x2", FLAGSHIP, "Gauss-iso", "Gauss-full", "GMM"):
        assert key in out.nlls, f"missing {key}"
        assert np.isfinite(out.nlls[key])
    assert out.modes[FLAGSHIP] in ("mixture", "single")


def test_flagship_never_worse_than_its_worst_member_by_much():
    """The gain gate must keep the flagship at/near the best member on val."""
    set_all(2)
    pip = FlowForgePipeline(TINY)
    from flowforge.data.preprocess import standardize_dataset
    from flowforge.data.synthetic import make_dataset

    ds = make_dataset("torus3", n_train=300, n_val=120, n_test=200, seed=2)
    ds, _ = standardize_dataset(ds)
    out = pip.run_one(ds, seed=2)
    members = [out.nlls["RealNVP-b6"], out.nlls["MAF-x2"]]
    assert out.nlls[FLAGSHIP] <= max(members) + 1e-6, "flagship regressed beyond worst member"


def test_benchmark_aggregates_over_seeds():
    set_all(3)
    pip = FlowForgePipeline(TINY)
    rep = pip.benchmark(["torus3"], [1, 2], verbose=False)
    assert len(rep.rows) >= 5
    for r in rep.rows:
        assert r.nll_std >= 0.0
        assert np.isfinite(r.nll_mean)
    assert rep.meta["seeds"] == [1, 2]


def test_strongest_baseline_is_min_nll():
    set_all(4)
    pip = FlowForgePipeline(TINY)
    rep = pip.benchmark(["torus3"], [1], verbose=False)
    best = FlowForgePipeline.strongest_baseline(rep.rows, "torus3")
    bases = [r for r in rep.rows if r.dataset == "torus3" and r.kind == "baseline"]
    assert best.nll_mean == min(r.nll_mean for r in bases)


def test_candidates_and_baselines_are_constructible():
    c = build_candidates(4, TINY, 0)
    b = build_baselines(4, TINY)
    assert len(c) == 2 and len(b) == 3
    assert all(hasattr(m, "log_prob") for _, m in c + b)


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("FLOWFORGE_EPOCHS", "37")
    cfg = Config()
    assert cfg.epochs == 37


def test_config_rejects_bad_schema():
    with pytest.raises(ConfigError):
        Config(epochs=0)
    with pytest.raises(ConfigError):
        Config(lr=5.0)
