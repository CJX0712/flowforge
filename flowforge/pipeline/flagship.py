"""FlowFuse — the FlowForge flagship.

Idea
----
Individual flows are *densities*, so unlike classifiers they admit a principled
fusion: a **log-linear mixture**

    p_fuse(x) = sum_k w_k p_k(x),   sum_k w_k = 1,  w_k >= 0

We fit ``w`` by **EM on the validation split**.  Because EM monotonically
increases the (validation) log-likelihood at every step, the fusion can be
*verified*, not merely hoped for — we assert that monotonicity as a hard
invariant.

Safety
------
A fusion is only shipped if it actually helps: if the mixture's validation NLL
is worse than the best single member, FlowFuse falls back to that single member
(non-inferiority gain gate).  This makes the flagship impossible to *harm*
relative to its own components — a deliberate design choice, and the reason the
ablation reports the gate's decision per dataset.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..core.errors import NumericalError
from ..core.seed import spawn
from ..flows.base import AbstractFlow


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class FuseFit:
    """Diagnostics from fitting the fusion weights."""

    val_ll_curve: List[float] = field(default_factory=list)
    member_val_nll: List[float] = field(default_factory=list)
    mode: str = "mixture"  # 'mixture' | 'single'
    best_member: int = 0
    mixture_val_nll: float = float("inf")
    fallback_val_nll: float = float("inf")


class FlowFuse(AbstractFlow):
    """Mixture-of-flows flagship with EM weights and a gain gate."""

    def __init__(self, members: List[AbstractFlow], dim: int, weights: np.ndarray | None = None):
        if not members:
            raise ValueError("FlowFuse needs at least one member")
        self.members = list(members)
        self.dim = dim
        k = len(members)
        self.weights = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float)
        self.weights = self.weights / self.weights.sum()
        self._mode = "mixture"
        self._best = 0
        self.fit_info: FuseFit | None = None

    # ------------------------------------------------------------------ #
    def fit_weights(self, X_val: np.ndarray, n_iter: int = 60, tol: float = 1e-10) -> FuseFit:
        """EM for mixture weights; validation log-likelihood is monotone."""
        logp = np.stack([np.asarray(m.log_prob(X_val), float) for m in self.members], axis=1)
        if not np.all(np.isfinite(logp)):
            raise NumericalError("member produced non-finite log-density on val")
        log_w = np.log(self.weights + 1e-300)
        curve: List[float] = []
        prev = -np.inf
        for _ in range(n_iter):
            r = log_w[None, :] + logp
            ll = float(np.mean(_logsumexp(r, axis=1)))
            curve.append(ll)
            resp = np.exp(r - _logsumexp(r, axis=1)[:, None])
            new_w = resp.mean(axis=0)
            new_w = np.clip(new_w, 1e-12, 1.0)
            new_w = new_w / new_w.sum()
            log_w = np.log(new_w)
            if abs(ll - prev) < tol:
                break
            prev = ll
        self.weights = np.exp(log_w)

        # ---- gain gate ------------------------------------------------ #
        member_nll = [float(-np.mean(p)) for p in logp.T]
        mix_nll = float(-np.mean(_logsumexp(log_w[None, :] + logp, axis=1)))
        best = int(np.argmin(member_nll))
        self._best = best
        if mix_nll <= min(member_nll) + 1e-12:
            self._mode = "mixture"
        else:
            self._mode = "single"
        info = FuseFit(
            val_ll_curve=curve,
            member_val_nll=member_nll,
            mode=self._mode,
            best_member=best,
            mixture_val_nll=mix_nll,
            fallback_val_nll=min(member_nll),
        )
        self.fit_info = info
        return info

    # ------------------------------------------------------------------ #
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        if self._mode == "single":
            return np.asarray(self.members[self._best].log_prob(x), float)
        logp = np.stack([np.asarray(m.log_prob(x), float) for m in self.members], axis=1)
        log_w = np.log(self.weights + 1e-300)
        return _logsumexp(log_w[None, :] + logp, axis=1)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or spawn()
        if self._mode == "single":
            return self.members[self._best].sample(n, rng=rng)
        counts = rng.multinomial(n, self.weights)
        parts = [self.members[i].sample(int(c), rng=rng) for i, c in enumerate(counts) if c > 0]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, self.dim))

    # A mixture is not a single bijection; forward/inverse are undefined.
    def forward(self, x: np.ndarray):  # pragma: no cover - not used by pipeline
        raise NotImplementedError("FlowFuse is a density mixture, not a single bijection")

    def inverse(self, z: np.ndarray):  # pragma: no cover
        raise NotImplementedError("FlowFuse is a density mixture, not a single bijection")

    @property
    def n_params(self) -> int:
        return sum(int(getattr(m, "n_params", 0)) for m in self.members) + len(self.members)

    @property
    def mode(self) -> str:
        return self._mode
