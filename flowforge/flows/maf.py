"""Masked Autoregressive Flow (MAF, Papamakarios et al., 2017) in pure NumPy.

MAF parameterises the *density* direction as an autoregressive model

    z_i = (x_i - mu_i(x_<i)) / exp(logs_i(x_<i)),
    log|det ∂z/∂x| = - sum_i logs_i(x_<i)

which is cheap to evaluate (one masked-MLP pass) but sequential to invert —
the classic MAF trade-off (fast density, slow sampling).

Autoregressive correctness ("reach" scheme)
-------------------------------------------
We assign every unit a *reach* ``r`` = the largest source coordinate whose
information it may carry, and connect layer ``l -> l+1`` only when
``r_dst >= r_src``.  With input reach ``r_in[c] = c`` and output reach
``r_out[c] = c - 1``, output coordinate ``c`` can only ever see inputs
``<= c-1``, i.e. exactly ``x_<c``.  Coordinate 0 gets reach ``-1`` and is
therefore unconditional (a *correct* constant head, matching the true
marginal).  This scheme is valid even for ``D = 2``, where the usual
"sample hidden order in [1, D-1]" recipe degenerates.

Author: 晨星
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.seed import spawn
from .base import AbstractFlow
from .composite import CompositeFlow

_LOG_CLIP = 7.0


class MaskedMLP:
    """MLP whose weight matrices are masked to be autoregressive."""

    def __init__(
        self,
        dim: int,
        d_out: int,
        hidden: int = 64,
        n_layers: int = 2,
        out_scale: float = 1.0,
        seed: int | None = None,
    ):
        self.dim = dim
        self.d_out = d_out
        self.hidden = hidden
        self.n_layers = max(2, n_layers)
        self.params: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}
        self._cache: dict = {}
        rng = spawn(seed)
        self._build(rng, out_scale)

    # ------------------------------------------------------------------ #
    def _build(self, rng: np.random.Generator, out_scale: float) -> None:
        sizes = [self.dim] + [self.hidden] * (self.n_layers - 1) + [self.d_out]
        self.sizes = sizes
        reaches: list[np.ndarray] = []
        reaches.append(np.arange(self.dim, dtype=np.int64))  # input reach = coord
        hi = max(self.dim - 2, 0)
        for _ in range(self.n_layers - 1):
            r = rng.integers(0, hi + 1, size=self.hidden).astype(np.int64)
            # force span: guarantee units that can see coord 0 and coordinate d-2
            if self.hidden >= 2:
                r[0] = 0
                r[-1] = hi
            reaches.append(r)
        coords = np.arange(self.d_out) % self.dim
        reaches.append((coords - 1).astype(np.int64))  # output reach = coord - 1

        for i in range(self.n_layers):
            r_src = reaches[i]
            r_dst = reaches[i + 1]
            # W has shape (n_src, n_dst); mask must match that orientation.
            mask = (r_dst[None, :] >= r_src[:, None]).astype(np.float64)
            scale = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            if i == self.n_layers - 1:
                scale *= out_scale
            w = rng.standard_normal((sizes[i], sizes[i + 1])) * scale
            b = np.zeros(sizes[i + 1])
            self.params[f"W{i}"] = w
            self.params[f"b{i}"] = b
            self.masks[f"W{i}"] = mask

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        cache = {"x0": x}
        for i in range(self.n_layers):
            w_eff = self.params[f"W{i}"] * self.masks[f"W{i}"]
            z = a @ w_eff + self.params[f"b{i}"]
            a = np.tanh(z) if i < self.n_layers - 1 else z
            cache[f"z{i}"] = z
            cache[f"a{i}"] = a
        self._cache = cache
        return a

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not self._cache:
            raise RuntimeError("MaskedMLP.backward before forward")
        c = self._cache
        grads: Dict[str, np.ndarray] = {}
        da = dout
        for i in reversed(range(self.n_layers)):
            a = c[f"a{i}"]
            # tanh'(z) = 1 - tanh(z)^2 = 1 - a^2   (a is the POST-activation)
            dz = da * (1.0 - a**2) if i < self.n_layers - 1 else da
            a_prev = c[f"a{i - 1}"] if i > 0 else c["x0"]
            gw = a_prev.T @ dz
            gw = gw * self.masks[f"W{i}"]  # masked connections carry no gradient
            grads[f"W{i}"] = gw
            grads[f"b{i}"] = dz.sum(axis=0)
            w_eff = self.params[f"W{i}"] * self.masks[f"W{i}"]
            da = dz @ w_eff.T
        return da, grads

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return sum(int(v.size) for v in self.params.values())

    def get_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            self.params[k] = np.asarray(v, dtype=np.float64)


class MAFStack(CompositeFlow):
    """Stacked MAF blocks separated by permutations.

    A *single* MAF layer models the first coordinate's marginal as a plain
    Gaussian (a consequence of autoregressive conditioning: coordinate 0 has no
    parents).  Stacking with permutations between blocks lets later blocks
    re-shape that marginal, which is what makes MAF competitive on multi-modal
    data.

    Note the classic MAF trade-off: density evaluation is one masked pass per
    block (fast), while sampling is sequential over coordinates.
    """

    def __init__(
        self,
        dim: int,
        n_blocks: int = 2,
        hidden: int = 64,
        n_layers: int = 2,
        seed: int | None = None,
        name: str = "maf-stack",
    ):
        from .permute import RandomPermutation

        blocks = []
        base = int(spawn(seed).integers(1 << 30)) if seed is not None else 0
        for b in range(n_blocks):
            blocks.append(MaskedAutoregressiveFlow(dim, hidden=hidden, n_layers=n_layers, seed=base + b))
            blocks.append(RandomPermutation(dim, seed=base + 500 + b))
        super().__init__(dim, blocks)
        self.name = name


class MaskedAutoregressiveFlow(AbstractFlow):
    """MAF: density-fast, sampling-sequential autoregressive flow."""

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        n_layers: int = 2,
        seed: int | None = None,
        name: str = "maf",
    ):
        self.dim = dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.name = name
        self.made = MaskedMLP(dim, 2 * dim, hidden=hidden, n_layers=n_layers, seed=seed)
        self._cache: dict = {}

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.made.forward(x)
        raw_mu = out[:, : self.dim]
        raw_logs = out[:, self.dim :]
        mu = raw_mu
        logs = np.clip(raw_logs, -_LOG_CLIP, _LOG_CLIP)
        inv = np.exp(-logs)
        z = (x - mu) * inv
        log_det = -np.sum(logs, axis=1)
        self._cache = dict(x=x, mu=mu, logs=logs, inv=inv, raw_logs=raw_logs)
        return z, log_det

    def inverse(self, z: np.ndarray) -> np.ndarray:
        x = np.zeros_like(z)
        for i in range(self.dim):
            out = self.made.forward(x)
            mu = out[:, i]
            logs = np.clip(out[:, self.dim + i], -_LOG_CLIP, _LOG_CLIP)
            x[:, i] = mu + np.exp(logs) * z[:, i]
        return x

    # ------------------------------------------------------------------ #
    def backward(self, dz: np.ndarray, dld: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        c = self._cache
        x, mu, inv = c["x"], c["mu"], c["inv"]
        d_mu = -inv * dz
        d_logs = -(x - mu) * inv * dz - np.asarray(dld, dtype=np.float64)[:, None]
        # straight-through clipping: zero gradient where |raw| hit the clip
        raw = c["raw_logs"]
        keep = (np.abs(raw) < _LOG_CLIP).astype(np.float64)
        dout = np.concatenate([d_mu, d_logs * keep], axis=1)
        dx_made, grads = self.made.backward(dout)
        dx = dz * inv + dx_made  # direct ∂z_i/∂x_i plus autoregressive path
        grads = {f"made.{k}": v for k, v in grads.items()}
        return dx, grads

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return self.made.n_params

    def get_params(self) -> Dict[str, np.ndarray]:
        return self.made.get_params()

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.made.set_params(params)

    def clip_grads(self, grads: Dict[str, np.ndarray], max_norm: float = 5.0) -> float:
        tot = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads.values())))
        if tot > max_norm and tot > 0:
            s = max_norm / tot
            for k in grads:
                grads[k] *= s
        return tot
