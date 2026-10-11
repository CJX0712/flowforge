"""Maximum-likelihood training for NumPy flows (Adam + early stopping).

The gradient of the negative log-likelihood

    L = -mean_i [ log p_base(f(x_i)) + log|det ∂f/∂x| ]

is threaded backwards with the shared contract: upstream gradients enter as
``dz = +z / N`` (from the base density) and ``dlog_det = -1 / N``, and every
block contributes its own analytic Jacobian term.

Determinism: batch shuffling uses a seeded ``Generator``; early stopping uses
the validation split only (never the test split).

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..core.errors import NumericalError, TrainingError
from ..core.seed import spawn
from ..flows.base import base_logprob
from ..flows.realnvp import param_pointers


def collect_pointers(model) -> Dict[str, Tuple[dict, str]]:
    """Map gradient keys -> mutable ``(container_dict, subkey)`` references."""
    out: Dict[str, Tuple[dict, str]] = {}
    if hasattr(model, "blocks"):  # CompositeFlow / RealNVP
        for i, blk in enumerate(model.blocks):
            for k, ref in param_pointers(blk).items():
                out[f"b{i}.{k}"] = ref
    elif hasattr(model, "made"):  # MAF
        for k in model.made.params:
            out[f"made.{k}"] = (model.made.params, k)
    return out


class Adam:
    """Parameter-keyed Adam with decoupled weight decay."""

    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(
        self,
        pointers: Dict[str, Tuple[dict, str]],
        grads: Dict[str, np.ndarray],
        weight_decay: float = 0.0,
    ) -> None:
        self.t += 1
        for key, (container, sub) in pointers.items():
            if key not in grads:
                continue
            g = np.asarray(grads[key], dtype=np.float64)
            p = container[sub]
            if weight_decay > 0.0:
                g = g + weight_decay * p
            if not np.all(np.isfinite(g)):
                raise NumericalError(f"non-finite gradient at {key}")
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g * g)
            mh = self.m[key] / (1 - self.b1**self.t)
            vh = self.v[key] / (1 - self.b2**self.t)
            container[sub] = p - self.lr * mh / (np.sqrt(vh) + self.eps)


@dataclass
class FitResult:
    """Outcome of one :func:`fit` call."""

    best_val_nll: float
    train_nll_first: float
    train_nll_last: float
    epochs_run: int
    val_curve: List[float] = field(default_factory=list)
    train_curve: List[float] = field(default_factory=list)


def _nll(model, x: np.ndarray) -> float:
    z, ld = model.forward(x)
    return float(-np.mean(base_logprob(z) + ld))


def fit(
    model,
    X: np.ndarray,
    X_val: np.ndarray,
    *,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    weight_decay: float = 0.0,
    patience: int = 60,
    seed: int = 0,
    grad_clip: float = 5.0,
    lr_decay: bool = True,
    verbose: bool = False,
) -> FitResult:
    """Train ``model`` by maximum likelihood with early stopping on ``X_val``."""
    pointers = collect_pointers(model)
    if not pointers:
        raise TrainingError("model exposes no trainable parameters")
    opt = Adam(lr=lr)
    rng = spawn(seed)
    n = X.shape[0]
    best = np.inf
    best_params = model.get_params() if hasattr(model, "get_params") else None
    epochs_run = 0
    val_curve: List[float] = []
    train_curve: List[float] = []
    first_train: float | None = None
    bad = 0

    for ep in range(epochs):
        if lr_decay and epochs > 1:
            # cosine annealing: avoids the late-stage oscillation of a fixed LR
            opt.lr = lr * 0.5 * (1.0 + np.cos(np.pi * ep / (epochs - 1)))
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x = X[idx]
            N = x.shape[0]
            z, ld = model.forward(x)
            logp = base_logprob(z) + ld
            if not np.all(np.isfinite(logp)):
                raise NumericalError("non-finite log-density during training")
            dz = z / N
            dld = -np.ones(N) / N
            _, grads = model.backward(dz, dld)
            tot = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads.values())))
            if tot > grad_clip and tot > 0:
                s = grad_clip / tot
                grads = {k: v * s for k, v in grads.items()}
            opt.step(pointers, grads, weight_decay=weight_decay)
        epochs_run += 1
        tr = _nll(model, X)
        va = _nll(model, X_val)
        train_curve.append(tr)
        val_curve.append(va)
        if first_train is None:
            first_train = tr
        if va < best - 1e-12:
            best = va
            if hasattr(model, "get_params"):
                best_params = model.get_params()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  ep{ep:4d} train_nll={tr:.4f} val_nll={va:.4f}")

    if best_params is not None and hasattr(model, "set_params"):
        model.set_params(best_params)
    last_train = _nll(model, X)
    if first_train is None:  # pragma: no cover
        first_train = last_train
    return FitResult(
        best_val_nll=float(best),
        train_nll_first=float(first_train),
        train_nll_last=float(last_train),
        epochs_run=epochs_run,
        val_curve=val_curve,
        train_curve=train_curve,
    )
