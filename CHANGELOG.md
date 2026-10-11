# Changelog

All notable changes to FlowForge. **Author: 晨星** · MIT.

---

## [0.1.0] — 2026-10-11

First release. Exact-likelihood normalizing-flow density estimation with the
`FlowFuse` mixture flagship.

### Added

**Flows (Tier-1, pure NumPy)**
- `AffineCoupling` — RealNVP coupling with exact algebraic inverse and
  exact `log|det| = Σ s`; single fused conditioner emitting `[s | t]`.
- `RandomPermutation` — `log|det| = 0` re-shuffle between blocks.
- `MaskedMLP` / `MaskedAutoregressiveFlow` / `MAFStack` — MAF with a provably
  correct "reach"-based autoregressive mask (valid at `D = 2`), stacked with
  permutations so later blocks can re-shape the unconditional first marginal.
- `RealNVP`, `CompositeFlow` — composition with additive log-determinants.
- Baselines: `IsotropicGaussian`, `FullCovGaussian`, `MixtureOfGaussians`
  (scikit-learn).

**Training**
- Analytic-gradient maximum likelihood, Adam, cosine LR annealing, gradient
  clipping, early stopping on validation with best-checkpoint restore.

**Flagship**
- `FlowFuse`: log-linear mixture of flows, weights fitted by **EM on the
  validation split** (monotone by the EM theorem), plus a non-inferiority gain
  gate that falls back to the best single member when the mixture is worse.

**Data**
- 10 synthetic DGPs; discriminative `d ≥ 3` suite (warp6/8, banana6/8, torus3)
  and a diagnostic 2-D suite. Structural randomness is separated from sampling
  randomness so every split shares one distribution.
- Leak-safe standardiser fitted on `X_train` only.

**Verification**
- 50 tests, 91% coverage; invariant suite (invertibility 4.4e-16, log-det
  exactness 0.0, gradient check 1.8e-10 / 7.8e-10, MAF autoregression,
  EM monotonicity, bit-identical determinism).
- CLI (`benchmark` / `fit` / `selftest`), demo writing `benchmark.json`,
  CI matrix 3.12/3.13, Dockerfile, Makefile, pinned lock.

### Result

Gate **4/5** discriminative datasets won vs. the strongest baseline;
fusion improves all 5; demo 73.4 s on CPU; determinism bit-identical.

### Known limitations

- `torus3` is a genuine loss (−9.89% vs GMM).
- `warp8` win (+0.77%) is narrow and fragile.
- 2-D toy datasets are non-discriminative — GMM wins all of them.

---

### Changelog of defects found and fixed during development

Recorded because each one is a genuine trap, not a typo:

| Defect | Root cause | Fix |
|---|---|---|
| Gradient check failed at 3.5e-2 | tanh derivative used the **pre**-activation `z` instead of the post-activation `a` | `1 − a²` |
| MADE mask shape mismatch | mask built as `(n_dst, n_src)` while `W` is `(n_src, n_dst)` | transpose the mask |
| Coupling backprop wrong | only the **s**-net's input gradient was propagated; the **t**-net's was discarded | add both paths |
| Conditioner shape error | full zero-padded vector passed to a net expecting only conditioning coords | explicit `cond_idx` selection |
| Huge train→test NLL gap | DGP drew its **structural** parameters from the sampling RNG → train/val/test came from three different distributions | fixed structural seed |
| banana DGP reached 1e56 | recursive `x_j = x_{j−1} + 0.8(x_{j−1}²−1)` compounds quadratically | depend on the latent, not the previous coordinate |
| RealNVP exploded to NLL 219 ± 333 | unbounded `exp(s)` let the Jacobian run away | clip `\|s\| ≤ 10` with straight-through gradient |
| Config env override crashed | `from __future__ import annotations` makes `Field.type` a *string* | resolve via `typing.get_type_hints` |
