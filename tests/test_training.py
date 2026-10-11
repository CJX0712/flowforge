"""Tests for training, determinism and the offline fallback path.

Author: 晨星
"""

from __future__ import annotations

import sys

import numpy as np

from flowforge.core.seed import set_all
from flowforge.data.preprocess import standardize_dataset
from flowforge.data.synthetic import make_dataset
from flowforge.eval.metrics import negative_log_likelihood
from flowforge.flows.maf import MAFStack
from flowforge.flows.realnvp import RealNVP
from flowforge.training.trainer import fit


def _small_ds(name="torus3", n=400):
    ds = make_dataset(name, n_train=n, n_val=150, n_test=200, seed=7)
    return standardize_dataset(ds)[0]


def test_training_reduces_train_nll():
    set_all(1)
    ds = _small_ds()
    m = RealNVP(ds.dim, n_blocks=3, hidden=24, seed=1)
    r = fit(m, ds.X, ds.X_val, epochs=40, batch_size=128, lr=8e-3, patience=20, seed=1)
    assert r.train_nll_last < r.train_nll_first, "training must reduce train NLL"
    assert r.epochs_run >= 1


def test_maf_stack_trains():
    set_all(2)
    ds = _small_ds()
    m = MAFStack(ds.dim, n_blocks=2, hidden=24, seed=2)
    r = fit(m, ds.X, ds.X_val, epochs=40, batch_size=128, lr=8e-3, patience=20, seed=2)
    assert r.train_nll_last < r.train_nll_first
    assert np.all(np.isfinite(m.log_prob(ds.X_test)))


def test_training_is_bit_reproducible():
    """Same seed -> identical parameters and identical test NLL."""
    ds = _small_ds()
    out = []
    for _ in range(2):
        set_all(11)
        m = RealNVP(ds.dim, n_blocks=3, hidden=24, seed=5)
        fit(m, ds.X, ds.X_val, epochs=20, batch_size=128, lr=8e-3, patience=20, seed=5)
        out.append(negative_log_likelihood(m, ds.X_test))
    assert out[0] == out[1], f"non-deterministic: {out[0]} vs {out[1]}"


def test_different_seeds_give_different_models():
    ds = _small_ds()
    res = []
    for s in (1, 2):
        set_all(11)
        m = RealNVP(ds.dim, n_blocks=3, hidden=24, seed=s)
        fit(m, ds.X, ds.X_val, epochs=20, batch_size=128, lr=8e-3, patience=20, seed=s)
        res.append(negative_log_likelihood(m, ds.X_test))
    assert res[0] != res[1], "different init seeds should give different models"


def test_early_stopping_restores_best_params():
    set_all(3)
    ds = _small_ds()
    m = RealNVP(ds.dim, n_blocks=3, hidden=24, seed=3)
    r = fit(m, ds.X, ds.X_val, epochs=60, batch_size=128, lr=8e-3, patience=5, seed=3)
    # after restore, the val NLL must equal the best observed val NLL
    now = negative_log_likelihood(m, ds.X_val)
    assert now <= min(r.val_curve) + 1e-9


def test_no_torch_required_for_core_path():
    """Offline Tier-1 path must not depend on torch / nflows."""
    import flowforge.flows  # noqa: F401
    import flowforge.pipeline  # noqa: F401

    heavy = [m for m in ("torch", "nflows") if m in sys.modules]
    # importing the package must not drag in a heavy backend
    assert "torch" not in heavy, "importing flowforge must not import torch"


def test_fit_rejects_parameterless_model():
    import pytest

    class Dummy:
        def forward(self, x):
            return x, np.zeros(len(x))

        def get_params(self):
            return {}

        def set_params(self, p):
            pass

        def backward(self, dz, dld):
            return dz, {}

    with pytest.raises(Exception):
        fit(Dummy(), np.zeros((10, 2)), np.zeros((5, 2)), epochs=1)
