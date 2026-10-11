"""Tests for leak-safe standardisation.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.seed import set_all, spawn
from flowforge.core.types import Dataset
from flowforge.data.preprocess import Standardizer, standardize_dataset
from flowforge.data.synthetic import (
    DISCRIMINATIVE,
    REGISTRY,
    dgp_warp,
    list_datasets,
    make_dataset,
)


def test_scaler_is_fitted_on_train_only():
    set_all(1)
    rng = spawn(1)
    ds = Dataset(
        "t",
        3,
        rng.normal(size=(100, 3)) + 5.0,
        rng.normal(size=(30, 3)) + 5.0,
        rng.normal(size=(30, 3)) + 5.0,
    )
    out, sc = standardize_dataset(ds)
    # the scaler's statistics must equal the TRAIN statistics, not the pooled ones
    assert np.allclose(sc.mean, ds.X.mean(axis=0))
    assert np.allclose(sc.std, ds.X.std(axis=0))
    # train is exactly standardised
    assert np.max(np.abs(out.X.mean(axis=0))) < 1e-12
    assert np.max(np.abs(out.X.std(axis=0) - 1.0)) < 1e-12


def test_log_abs_det_matches_manual():
    set_all(2)
    rng = spawn(2)
    X = rng.normal(size=(50, 4)) * np.array([1.0, 2.0, 3.0, 4.0])
    sc = Standardizer().fit(X)
    assert sc.log_abs_det == pytest.approx(-float(np.sum(np.log(sc.std))), rel=1e-12)


def test_transform_is_affine_and_invertible_in_spirit():
    set_all(3)
    rng = spawn(3)
    X = rng.normal(size=(40, 5))
    sc = Standardizer().fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z, (X - sc.mean) / sc.std)


def test_standardizer_requires_fit():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.zeros((2, 2)))


# --------------------------------------------------------------------- #
# DGP sanity: every split must come from the SAME distribution.
# (A structural parameter drawn from the sampling RNG would silently give
# train/val/test three different distributions.)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list_datasets())
def test_splits_share_one_distribution(name):
    set_all(4)
    _, dim = REGISTRY[name]
    ds = make_dataset(name, n_train=1500, n_val=800, n_test=1500, seed=99)
    for other in (ds.X_val, ds.X_test):
        # per-coordinate mean/std must agree within sampling error
        assert np.max(np.abs(ds.X.mean(0) - other.mean(0))) < 0.45, f"{name}: split means diverge"
        assert np.max(np.abs(ds.X.std(0) / other.std(0) - 1.0)) < 0.45, f"{name}: split scales diverge"


def test_warp_structure_is_fixed_across_calls():
    """Two independent calls must sample the SAME warp, not two different ones."""
    set_all(5)
    a = dgp_warp(500, spawn(11), dim=6)
    b = dgp_warp(500, spawn(12), dim=6)
    # different samples, identical distribution -> comparable moments
    assert np.max(np.abs(a.mean(0) - b.mean(0))) < 0.45
    assert np.max(np.abs(a.std(0) / b.std(0) - 1.0)) < 0.45


def test_discriminative_suite_is_registered():
    for name in DISCRIMINATIVE:
        assert name in REGISTRY, f"{name} missing from registry"
        assert REGISTRY[name][1] >= 3, "discriminative suite must be d>=3"
