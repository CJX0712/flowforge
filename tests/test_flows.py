"""Invariant tests: the mathematical guarantees FlowForge is built on.

These are *falsifiable* checks, not smoke tests — each one would fail loudly if
the change-of-variables mathematics were wrong.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from flowforge.core.errors import ConfigError
from flowforge.core.seed import set_all, spawn
from flowforge.flows.base import base_logprob
from flowforge.flows.composite import CompositeFlow
from flowforge.flows.coupling import AffineCoupling
from flowforge.flows.maf import MaskedAutoregressiveFlow
from flowforge.flows.permute import RandomPermutation
from flowforge.flows.realnvp import RealNVP, half_masks
from flowforge.training.trainer import collect_pointers as cp


def _numeric_logdet(model, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """log|det ∂z/∂x| by central differences of the forward map."""
    n, d = x.shape
    jac = np.zeros((n, d, d))
    for j in range(d):
        xp = x.copy()
        xp[:, j] += eps
        xm = x.copy()
        xm[:, j] -= eps
        zp = model.forward(xp)[0]
        zm = model.forward(xm)[0]
        jac[:, :, j] = (zp - zm) / (2 * eps)
    dets = np.linalg.det(jac)
    return np.log(np.abs(dets))


# --------------------------------------------------------------------------- #
def test_affine_coupling_inverse_is_exact():
    set_all(1)
    rng = spawn(1)
    mask = np.array([1.0, 0.0])
    blk = AffineCoupling(dim=2, mask=mask, hidden=16, seed=3)
    x = rng.normal(size=(32, 2))
    z, _ = blk.forward(x)
    assert np.max(np.abs(blk.inverse(z) - x)) < 1e-10


def test_affine_coupling_logdet_equals_sum_of_scale():
    """RealNVP invariant: log|det| == sum(s) exactly (no approximation)."""
    set_all(2)
    rng = spawn(2)
    blk = AffineCoupling(dim=2, mask=np.array([1.0, 0.0]), hidden=16, seed=5)
    x = rng.normal(size=(64, 2))
    _, ld = blk.forward(x)
    s = blk.net.forward(x[:, blk.cond_idx])[:, : blk.n_trans]
    assert np.max(np.abs(ld - np.sum(s, axis=1))) < 1e-12


def test_numeric_jacobian_matches_analytic_logdet():
    """Change of variables: analytic log|det| == finite-difference log|det|."""
    set_all(3)
    rng = spawn(3)
    model = RealNVP(dim=3, n_blocks=3, hidden=24, n_layers=2, seed=11)
    x = rng.normal(size=(8, 3)) * 0.5
    _, ld = model.forward(x)
    num = _numeric_logdet(model, x)
    assert np.max(np.abs(ld - num)) < 1e-5


def test_composite_logdet_is_sum_of_parts():
    set_all(4)
    rng = spawn(4)
    dim = 3
    masks = half_masks(dim)
    blocks = [
        AffineCoupling(dim, masks[0], hidden=16, seed=1),
        RandomPermutation(dim, seed=2),
        AffineCoupling(dim, masks[1], hidden=16, seed=3),
    ]
    comp = CompositeFlow(dim, blocks)
    x = rng.normal(size=(16, 3))
    _, total = comp.forward(x)
    parts = 0.0
    z = x
    for b in blocks:
        z, ld = b.forward(z)
        parts = parts + ld
    assert np.max(np.abs(total - parts)) < 1e-12


def test_permutation_roundtrip():
    set_all(5)
    rng = spawn(5)
    p = RandomPermutation(5, seed=9)
    x = rng.normal(size=(4, 5))
    z, ld = p.forward(x)
    assert np.allclose(ld, 0.0)
    assert np.max(np.abs(p.inverse(z) - x)) < 1e-15


def test_gradient_check_realnvp():
    """Analytic backprop vs central differences (the whole training signal)."""
    set_all(6)
    rng = spawn(6)
    model = RealNVP(dim=3, n_blocks=3, hidden=16, n_layers=2, seed=5)
    x = rng.normal(size=(7, 3))

    def loss():
        z, ld = model.forward(x)
        return float(-np.mean(base_logprob(z) + ld))

    z, _ = model.forward(x)
    n = x.shape[0]
    _, grads = model.backward(z / n, -np.ones(n) / n)
    pointers = cp(model)
    assert set(grads) == set(pointers), "gradient keys must match parameter pointers"

    worst = 0.0
    for key in list(pointers)[:8]:
        container, sub = pointers[key]
        base = container[sub].copy()
        idx = np.unravel_index(np.argmin(np.abs(base)) + 1, base.shape)
        eps = 1e-6
        container[sub][idx] = base[idx] + eps
        lp = loss()
        container[sub][idx] = base[idx] - eps
        lm = loss()
        container[sub][idx] = base[idx]
        num = (lp - lm) / (2 * eps)
        ana = float(grads[key][idx])
        worst = max(worst, abs(num - ana) / max(1.0, abs(num), abs(ana)))
    assert worst < 1e-6, f"gradient check failed, worst rel err {worst:.3e}"


def test_gradient_check_maf():
    set_all(7)
    rng = spawn(7)
    maf = MaskedAutoregressiveFlow(dim=4, hidden=24, n_layers=2, seed=3)
    x = rng.normal(size=(6, 4))

    def loss():
        z, ld = maf.forward(x)
        return float(-np.mean(base_logprob(z) + ld))

    z, _ = maf.forward(x)
    n = x.shape[0]
    _, grads = maf.backward(z / n, -np.ones(n) / n)
    pointers = cp(maf)
    worst = 0.0
    for key, (container, sub) in list(pointers.items())[:6]:
        base = container[sub].copy()
        idx = np.unravel_index(np.argmin(np.abs(base)) + 1, base.shape)
        eps = 1e-6
        container[sub][idx] = base[idx] + eps
        lp = loss()
        container[sub][idx] = base[idx] - eps
        lm = loss()
        container[sub][idx] = base[idx]
        num = (lp - lm) / (2 * eps)
        ana = float(grads[f"made.{sub}"][idx])
        worst = max(worst, abs(num - ana) / max(1.0, abs(num), abs(ana)))
    assert worst < 1e-6, f"MAF gradient check failed, worst rel err {worst:.3e}"


def test_maf_inverse_is_exact():
    set_all(8)
    rng = spawn(8)
    maf = MaskedAutoregressiveFlow(dim=4, hidden=32, n_layers=2, seed=3)
    x = rng.normal(size=(40, 4))
    z, _ = maf.forward(x)
    assert np.max(np.abs(maf.inverse(z) - x)) < 1e-9


def test_maf_is_strictly_autoregressive():
    """Output for coordinate c must not depend on x_{>=c}."""
    set_all(9)
    rng = spawn(9)
    dim = 5
    maf = MaskedAutoregressiveFlow(dim=dim, hidden=48, n_layers=3, seed=9)
    x = rng.normal(size=(3, dim))
    out0 = maf.made.forward(x)
    for c in range(dim):
        xp = x.copy()
        xp[:, c:] += 5.0
        out1 = maf.made.forward(xp)
        for j in range(2 * dim):
            if j % dim == c:
                assert np.max(np.abs(out0[:, j] - out1[:, j])) < 1e-12, f"coordinate {c} leaked into output {j}"


def test_coupling_mask_must_split_dimensions():
    set_all(10)
    with pytest.raises(ConfigError):
        AffineCoupling(dim=3, mask=np.ones(3), hidden=8, seed=1)


def test_sample_shape_and_finiteness():
    set_all(11)
    model = RealNVP(dim=2, n_blocks=2, hidden=16, seed=4)
    s = model.sample(25, rng=spawn(11))
    assert s.shape == (25, 2)
    assert np.all(np.isfinite(s))
