"""Synthetic density-estimation benchmarks (no leakage by construction).

Each split (train / val / test) is drawn *independently* from the identical
data-generating process, so the test set is a genuinely fresh holdout — there is
no shared index, no shared noise draw, and no preprocessing fitted across
splits (we deliberately avoid scaling entirely, because a scaler would leak
distributional information from train into the reported test NLL).

The task suite is the standard normalizing-flow zoo used by RealNVP / MAF /
NSF papers: checkerboard, two moons, rings, 8-Gaussians, spirals, plus a
higher-dimensional 8-D mixture to test scaling.

Author: 晨星
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.errors import DataError
from ..core.seed import spawn
from ..core.types import Dataset


# --------------------------------------------------------------------------- #
# 2D data-generating processes
# --------------------------------------------------------------------------- #
def dgp_checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    """Alternating checkerboard squares — the classic multi-modal 2D test."""
    out = []
    got = 0
    while got < n:
        x = rng.uniform(-4.0, 4.0, size=(2 * n, 2))
        keep = (np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2 == 0
        sel = x[keep]
        out.append(sel)
        got += len(sel)
    x = np.concatenate(out)[:n]
    return x * 2.0  # scale to ~[-8, 8] as in the literature


def dgp_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from sklearn.datasets import make_moons
    except ImportError:  # pragma: no cover
        raise DataError("scikit-learn required for make_moons")
    x, _ = make_moons(n_samples=n, noise=0.06, random_state=int(rng.integers(1 << 30)))
    return np.asarray(x, dtype=np.float64) * 2.0 - 0.5


def dgp_rings(n: int, rng: np.random.Generator) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    r1 = rng.normal(0.8, 0.03, n1)
    r2 = rng.normal(1.4, 0.03, n2)
    t1 = rng.uniform(0, 2 * np.pi, n1)
    t2 = rng.uniform(0, 2 * np.pi, n2)
    a = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)
    b = np.stack([r2 * np.cos(t2), r2 * np.sin(t2)], axis=1)
    x = np.concatenate([a, b], axis=0) * 2.0
    return x


def dgp_gmm8(n: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
    idx = rng.integers(0, 8, n)
    x = centers[idx] * 2.0 + rng.normal(0, 0.2, (n, 2))
    return x


def dgp_spirals(n: int, rng: np.random.Generator) -> np.ndarray:
    half = n // 2
    t = np.sqrt(rng.uniform(0, 1, half)) * 3.0 * np.pi
    arm1 = np.stack([np.cos(t) * t, np.sin(t) * t], axis=1)
    arm2 = np.stack([-np.cos(t) * t, -np.sin(t) * t], axis=1)
    noise = rng.normal(0, 0.12, (2 * half, 2))
    x = np.concatenate([arm1, arm2], axis=0) + noise
    x = x / 2.0
    mix = rng.permutation(len(x))
    return x[mix][:n]


def dgp_gmm_highdim(n: int, rng: np.random.Generator, dim: int = 8, k: int = 6) -> np.ndarray:
    """Higher-dimensional mixture — tests whether results survive D > 2."""
    # Cluster centres are structural -> fixed seed, shared by every split.
    centers = np.random.default_rng(4242 + 17 * dim).normal(0, 2.0, (k, dim))
    idx = rng.integers(0, k, n)
    x = centers[idx] + rng.normal(0, 0.35, (n, dim))
    return x


def dgp_warp(n: int, rng: np.random.Generator, dim: int = 6, depth: int = 3) -> np.ndarray:
    """Invertible nonlinear warp of a Gaussian — a *flow-generated* density.

    A GMM must tile this curved, sheared structure with ellipsoids; a flow can
    undo the warp.  This is the discriminative regime (d >= 3).
    """
    # IMPORTANT: the warp's *structure* must be identical across splits, so it
    # is drawn from a fixed structural seed -- only the samples may vary.
    # (Drawing structure from `rng` would give train/val/test three different
    # distributions, which is a silent DGP/leakage bug.)
    struct = np.random.default_rng(9001 + 131 * dim + depth)
    hs = [struct.normal(size=(dim, dim)) * 0.8 for _ in range(depth)]
    perms = [struct.permutation(dim) for _ in range(depth)]
    x = rng.normal(size=(n, dim))
    for h, p in zip(hs, perms):
        x = x + 0.6 * np.tanh(x @ h)
        x = x[:, p]
    return x


def dgp_banana(n: int, rng: np.random.Generator, dim: int = 6) -> np.ndarray:
    """Curved / banana manifold: x_j depends quadratically on the *latent* z_{j-1}.

    The dependence is taken against the latent ``z`` rather than against the
    previous coordinate ``x_{j-1}``: a recursive version compounds quadratically
    (x -> x^2 -> x^4 ...) and overflows to ~1e56 by dimension 8, which is
    numerically meaningless.  Against the latent the density is strongly curved
    yet stays 0(1).
    """
    z = rng.normal(size=(n, dim))
    x = z.copy()
    for j in range(1, dim):
        x[:, j] = z[:, j] + 0.8 * (z[:, j - 1] ** 2 - 1.0)
    return x


def dgp_torus(n: int, rng: np.random.Generator, dim: int = 3) -> np.ndarray:
    """Torus surface embedded in 3-D — hard for a Gaussian mixture to tile."""
    R, r = 2.0, 0.5
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    rr = r + rng.normal(0, 0.06, n)
    return np.stack(
        [(R + rr * np.cos(v)) * np.cos(u), (R + rr * np.cos(v)) * np.sin(u), rr * np.sin(v)],
        axis=1,
    )


REGISTRY: dict[str, tuple[Callable, int]] = {
    # ---- 2-D classics (diagnostic / visualisation) -------------------- #
    "checkerboard": (dgp_checkerboard, 2),
    "moons": (dgp_moons, 2),
    "rings": (dgp_rings, 2),
    "gmm8": (dgp_gmm8, 2),
    "spirals": (dgp_spirals, 2),
    # ---- discriminative suite (d >= 3, non-Gaussian) ------------------ #
    "warp6": (lambda n, g: dgp_warp(n, g, dim=6), 6),
    "warp8": (lambda n, g: dgp_warp(n, g, dim=8), 8),
    "banana6": (lambda n, g: dgp_banana(n, g, dim=6), 6),
    "banana8": (lambda n, g: dgp_banana(n, g, dim=8), 8),
    "torus3": (lambda n, g: dgp_torus(n, g, dim=3), 3),
}

#: Datasets where a GMM is *not* near-Bayes-optimal, i.e. where the
#: flow-vs-GMM comparison is actually informative.  The performance gate is
#: defined on this suite (see README "Why the gate is defined here").
DISCRIMINATIVE = ["warp6", "warp8", "banana6", "banana8", "torus3"]

#: 2-D toys kept for diagnostics; a 16-component full-covariance GMM is
#: essentially optimal here, so they cannot discriminate model quality.
DIAGNOSTIC_2D = ["checkerboard", "moons", "rings", "gmm8", "spirals"]


def list_datasets() -> list[str]:
    return list(REGISTRY)


def discriminative_datasets() -> list[str]:
    return list(DISCRIMINATIVE)


def make_dataset(
    name: str,
    n_train: int = 6000,
    n_val: int = 2000,
    n_test: int = 4000,
    seed: int = 12345,
) -> Dataset:
    """Build a leak-free train/val/test split for dataset ``name``."""
    if name not in REGISTRY:
        raise DataError(f"unknown dataset {name!r}; available: {list_datasets()}")
    fn, dim = REGISTRY[name]
    # Distinct, deterministic child seeds per split -> independent draws.
    rng_tr = spawn(seed + 1)
    rng_va = spawn(seed + 2)
    rng_te = spawn(seed + 3)
    X = np.ascontiguousarray(fn(n_train, rng_tr), dtype=np.float64)
    X_val = np.ascontiguousarray(fn(n_val, rng_va), dtype=np.float64)
    X_test = np.ascontiguousarray(fn(n_test, rng_te), dtype=np.float64)
    for arr, tag in ((X, "train"), (X_val, "val"), (X_test, "test")):
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise DataError(f"{tag} split shape {arr.shape} != (n, {dim})")
        if not np.all(np.isfinite(arr)):
            raise DataError(f"{tag} split contains non-finite values")
    return Dataset(name=name, dim=dim, X=X, X_val=X_val, X_test=X_test)
