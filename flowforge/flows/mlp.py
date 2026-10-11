"""A tiny, dependency-free MLP with analytic forward + reverse-mode backward.

Every flow conditioner (affine coupling, MADE, spline net) is just an MLP, so a
single well-tested backprop implementation backs the whole library.  Weights
are initialised with a controllable output scale so that, e.g. the *scale* head
of a coupling starts near zero (``exp(s) ≈ 1``) — a known RealNVP stability
trick (Dinh et al., 2016).

Determinism: initialisation uses a ``numpy`` Generator seeded from the global
seed via :func:`core.seed.spawn`.

Author: 晨星
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.errors import NumericalError
from ..core.seed import spawn


class MLP:
    """2-hidden-layer MLP: ``tanh`` activations, linear output head.

    Shape of parameters per layer ``l``: ``W_l`` (in, out), ``b_l`` (out,).
    ``out_scale`` multiplies the *final* weight matrix so we can start a head
    near zero without harming the hidden layers.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int = 64,
        n_layers: int = 2,
        out_scale: float = 1.0,
        seed: int | None = None,
        name: str = "mlp",
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.hidden = hidden
        self.n_layers = max(1, n_layers)
        self.out_scale = out_scale
        self.name = name
        self._cache: dict = {}
        self._build(seed)

    # ------------------------------------------------------------------ #
    def _build(self, seed: int | None) -> None:
        rng = spawn(seed)
        self.params: Dict[str, np.ndarray] = {}
        self._order: List[str] = []
        sizes = [self.d_in] + [self.hidden] * (self.n_layers - 1) + [self.d_out]
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            # Xavier-ish for hidden; scaled for the output head.
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            if i == len(sizes) - 2:  # last layer
                scale *= self.out_scale
            w = rng.standard_normal((fan_in, fan_out)) * scale
            b = np.zeros(fan_out)
            self.params[f"W{i}"] = w
            self.params[f"b{i}"] = b
            self._order.append(f"W{i}")
            self._order.append(f"b{i}")

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x (N, d_in) -> out (N, d_out).  Caches activations for backward."""
        a = x
        cache = {"x0": x}
        for i in range(self.n_layers):  # n_layers hidden transitions + output
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < self.n_layers - 1:  # hidden layers use tanh
                a = np.tanh(z)
            else:  # final layer linear
                a = z
            cache[f"z{i}"] = z
            cache[f"a{i}"] = a
        self._cache = cache
        return a

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Reverse pass.  ``dout`` (N, d_out) -> (dx (N,d_in), grads dict)."""
        if not self._cache:
            raise NumericalError(f"{self.name}: backward called before forward")
        cache = self._cache
        x0 = cache["x0"]
        grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}
        # walk backwards over transitions: layers n_layers-1 .. 0
        da = dout
        for i in reversed(range(self.n_layers)):
            a = cache[f"a{i}"]
            if i < self.n_layers - 1:
                # tanh'(z) = 1 - tanh(z)^2 = 1 - a^2  (a is the POST-activation)
                dz = da * (1.0 - a**2)
            else:
                dz = da  # linear
            a_prev = cache[f"a{i - 1}"] if i > 0 else x0
            grads[f"W{i}"] = a_prev.T @ dz
            grads[f"b{i}"] = dz.sum(axis=0)
            da = dz @ self.params[f"W{i}"].T
        dx = da  # after i==0, da is gradient w.r.t x0
        return dx, grads

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return sum(int(v.size) for v in self.params.values())

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            self.params[k] = np.asarray(v, dtype=np.float64)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def clip_grads(self, grads: Dict[str, np.ndarray], max_norm: float = 5.0) -> float:
        tot = np.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
        if tot > max_norm and tot > 0:
            scale = max_norm / tot
            for k in grads:
                grads[k] *= scale
        return float(tot)
