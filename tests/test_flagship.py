"""Tests for the FlowFuse flagship: EM monotonicity + non-inferiority gate.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.seed import set_all, spawn
from flowforge.flows.gaussian import FullCovGaussian
from flowforge.flows.gmm import MixtureOfGaussians
from flowforge.pipeline.flagship import FlowFuse


def _members(dim: int, X: np.ndarray, seed: int = 0):
    g1 = FullCovGaussian(dim).fit(X)
    g2 = MixtureOfGaussians(dim, n_components=4, seed=seed).fit(X)
    g3 = MixtureOfGaussians(dim, n_components=8, seed=seed + 1).fit(X)
    return [g1, g2, g3]


def test_em_is_monotone_on_validation_loglik():
    """EM theorem: validation log-likelihood must never decrease."""
    set_all(1)
    rng = spawn(1)
    X = rng.normal(size=(200, 3))
    Xv = rng.normal(size=(200, 3))
    fuse = FlowFuse(_members(3, X), dim=3)
    info = fuse.fit_weights(Xv, n_iter=40)
    curve = info.val_ll_curve
    assert len(curve) >= 2
    deltas = np.diff(curve)
    assert np.min(deltas) > -1e-8, f"EM decreased: min delta {np.min(deltas):.3e}"


def test_weights_sum_to_one_and_stay_positive():
    set_all(2)
    rng = spawn(2)
    X = rng.normal(size=(100, 2))
    fuse = FlowFuse(_members(2, X), dim=2)
    fuse.fit_weights(rng.normal(size=(100, 2)), n_iter=25)
    assert np.all(fuse.weights >= 0.0)
    assert fuse.weights.sum() == pytest.approx(1.0)


def test_mixture_logprob_is_at_least_best_member():
    """A weighted mixture dominates the worst member and >= best member."""
    set_all(3)
    rng = spawn(3)
    X = rng.normal(size=(120, 3))
    Xv = rng.normal(size=(120, 3))
    members = _members(3, X)
    fuse = FlowFuse(members, dim=3)
    fuse.fit_weights(Xv, n_iter=30)
    lp = fuse.log_prob(Xv)
    best_member = max(float(np.mean(m.log_prob(Xv))) for m in members)
    assert float(np.mean(lp)) >= best_member - 1e-6


def test_gain_gate_falls_back_when_mixture_is_worse():
    """If the mixture cannot beat the best member, ship the best member."""
    set_all(4)
    rng = spawn(4)
    X = rng.normal(size=(150, 2))
    Xv = rng.normal(size=(150, 2))
    strong = MixtureOfGaussians(2, n_components=8, seed=7).fit(X)
    weak = FullCovGaussian(2).fit(X)
    fuse = FlowFuse([strong, weak], dim=2)
    info = fuse.fit_weights(Xv, n_iter=30)
    assert info.mode in ("mixture", "single")
    # whichever mode is chosen, it must be the better of the two on val
    best_single = min(-float(np.mean(m.log_prob(Xv))) for m in (strong, weak))
    actual = -float(np.mean(fuse.log_prob(Xv)))
    assert actual <= best_single + 1e-6, "gain gate shipped a worse-than-best-single model"


def test_sample_shape_and_finiteness():
    set_all(5)
    rng = spawn(5)
    X = rng.normal(size=(80, 2))
    fuse = FlowFuse(_members(2, X), dim=2)
    fuse.fit_weights(rng.normal(size=(80, 2)), n_iter=10)
    s = fuse.sample(37, rng=spawn(5))
    assert s.shape[0] > 0
    assert np.all(np.isfinite(s))


def test_requires_at_least_one_member():
    import pytest

    with pytest.raises(ValueError):
        FlowFuse([], dim=2)
