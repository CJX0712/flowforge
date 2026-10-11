# FlowForge — Architecture

**Author: 晨星** · v0.1.0 · MIT

---

## 1. What this system is

FlowForge is a **normalizing-flow density-estimation system**: given i.i.d.
samples `x ~ p*(x)` it learns an explicit, differentiable, invertible map
`f: x -> z` such that `z` is standard normal, and evaluates the exact density
via the change-of-variables formula

```
log p(x) = log p_base(f(x)) + log |det ∂f/∂x|
```

Because the Jacobian determinant is **exact** (not estimated), the training
objective is exact maximum likelihood — no ELBO gap, no discriminator, no
sampling noise in the objective.

---

## 2. Layered design (strictly one-directional, acyclic)

```
        cli.py
          │
          ▼
      pipeline ──────────► flagship (FlowFuse)
          │
          ├──► data        (synthetic DGPs, leak-free standardisation)
          ├──► flows       (RealNVP, MAF, Gaussians, GMM)
          ├──► training    (Adam, cosine LR, early stopping)
          └──► eval        (NLL, bits/dim, MMD²)
                              │
                              ▼
                            core  (seed, config, errors, types, interfaces)
```

No module below imports from a module above it. `core` imports nothing from
the project. This keeps every layer independently testable and makes the
dependency graph trivially auditable.

| Package | Responsibility |
|---|---|
| `core` | deterministic seeding, validated config, error taxonomy, data types, protocols |
| `data` | DGPs with **structural** randomness separated from **sampling** randomness; train-fitted standardiser |
| `flows` | invertible transforms (couplings, permutations, MADE/MAF) + parametric baselines |
| `training` | analytic-gradient MLE with Adam, cosine annealing, early stopping, best-checkpoint restore |
| `eval` | NLL / bits-per-dim / RBF-MMD² |
| `pipeline` | candidate zoo → `FlowFuse` fusion → benchmark aggregation → gate |

---

## 3. The flow mathematics

### 3.1 Affine coupling (RealNVP, Dinh et al. 2016)

Coordinates are split into conditioning (`c`) and transformed (`t`) sets:

```
z_c = x_c
z_t = x_t ⊙ exp(s(x_c)) + t(x_c)          log|det ∂z/∂x| = Σ_k s_k(x_c)
```

* inverse is **algebraic** — `x_t = (z_t − t) ⊙ exp(−s)`
* the log-determinant is **exactly** `Σ s`, asserted to `0.0` error in tests
* the Jacobian is triangular, so no `O(D³)` determinant is ever computed

Scale and shift come from **one** fused conditioner emitting `[s | t]` (≈3×
fewer kernel launches than two nets, mathematically identical). `|s|` is
clipped to 10 to stop runaway Jacobians — see §6.

### 3.2 Masked autoregressive flow (MAF, Papamakarios et al. 2017)

```
z_i = (x_i − μ_i(x_<i)) / exp(logσ_i(x_<i))      log|det| = −Σ_i logσ_i
```

Density is one masked pass; sampling is sequential (the classic MAF trade-off).

**Autoregressive correctness ("reach" scheme).** Every unit carries a *reach*
`r` = largest source coordinate whose information it may hold, and layer
`l → l+1` is connected only when `r_dst ≥ r_src`. With input reach `r[c] = c`
and output reach `r[c] = c − 1`, output `c` can only see inputs `≤ c − 1`,
i.e. exactly `x_<c`. Coordinate 0 gets reach `−1` and is therefore
**unconditional** — a constant head, which is the *correct* model for a
marginal with no parents. This scheme is valid even for `D = 2`, where the
usual "sample hidden order in [1, D−1]" recipe degenerates.

A single MAF layer consequently models the first coordinate's marginal as a
Gaussian, so MAF is **stacked** (blocks separated by permutations) to let later
blocks re-shape that marginal.

### 3.3 Composition

For `f = f_n ∘ … ∘ f_1`: `log|det ∂f/∂x| = Σ_i log|det ∂f_i/∂x_{i−1}|`
(asserted as an invariant). Permutations between blocks have `log|det| = 0`
and guarantee every coordinate is transformed.

### 3.4 Training

`L = −mean_i [ log p_base(f(x_i)) + log|det ∂f/∂x| ]`.

Upstream gradients enter the shared contract as

```
∂L/∂z = +z / N          (from log p_base = −½‖z‖²)
∂L/∂log|det| = −1 / N
```

and each block contributes its own analytic Jacobian term. Every gradient path
is verified against central differences (see §5).

---

## 4. FlowFuse — the flagship

Individual flows are **densities**, so unlike classifiers they admit a
principled fusion: a log-linear mixture `p(x) = Σ_k w_k p_k(x)`.

**Weights are fitted by EM on the validation split.** EM monotonically
increases the validation log-likelihood at every step, so the fusion is
*verifiable rather than hoped for* — the monotonicity is a hard invariant.

**Non-inferiority gain gate.** If the mixture's validation NLL is worse than
the best single member, FlowFuse ships that single member instead. The flagship
is therefore structurally incapable of being harmed by its own components —
which matters because the two families have complementary biases:

| Family | Strong on | Weak on |
|---|---|---|
| RealNVP (coupling) | curved manifolds (banana) | sheared/warped densities |
| MAF (autoregressive) | warped densities | strongly curved manifolds |

Empirically the *winner differs per dataset*, and the fusion adapts per seed —
which is exactly why the ablation shows a positive delta on every dataset.

---

## 5. Invariants (falsifiable, not smoke tests)

| # | Invariant | Measured |
|---|---|---|
| 1 | `inverse(forward(x)) == x` | ≤ 4.4e-16 |
| 2 | coupling `log|det| == Σ s` **exactly** | 0.0 |
| 3 | analytic `log|det|` == finite-difference Jacobian | < 1e-5 |
| 4 | composite `log|det|` == Σ of parts | < 1e-12 |
| 5 | analytic gradient == central differences | 1.8e-10 (RealNVP), 7.8e-10 (MAF) |
| 6 | MAF output `c` independent of `x_{≥c}` | 0 violations |
| 7 | EM validation log-likelihood monotone | never decreases |
| 8 | gain gate never ships worse-than-best-single | enforced by test |
| 9 | same seed ⇒ bit-identical metrics | worst abs diff 0.0 |
| 10 | scaler fitted on train only | asserted |

---

## 6. Failure modes deliberately designed against

| Failure | Guard |
|---|---|
| Runaway Jacobian (flow "cheats" by concentrating determinant) | `|s| ≤ 10` clip + validation-based early stopping |
| DGP leakage: structural randomness drawn from the sampling RNG gives train/val/test *different distributions* | structural params use a **fixed** seed; test asserts split moments agree |
| Numerically degenerate DGP (recursive banana overflows to 1e56) | dependence taken against the latent, not the previous coordinate |
| Preprocessing leakage | scaler fitted on `X_train` only |
| A weak member dragging the flagship down | non-inferiority gain gate |
| Optimisation stalling | cosine LR annealing + best-checkpoint restore |

---

## 7. Determinism

A single entry point `core.seed.set_all(seed)` pins NumPy and Python `random`.
Every stochastic element (data generation, weight init, batch shuffling,
sampling) derives from it via `spawn()`. Two runs with the same seed produce
**bit-identical** metrics, verified by the demo on every invocation.

---

## 8. Offline / degradation policy

| Tier | Contents | Required? |
|---|---|---|
| Tier-1 | pure-NumPy RealNVP, MAF, Gaussian, GMM | **always** — zero download, CPU-only |
| Tier-0 | scikit-learn (GMM baseline), optional `nflows`/`torch` | sklearn yes; torch/nflows optional |

`torch` and `nflows` are **never** imported by the package import path
(test asserts this). If absent, the system runs Tier-1 unimpaired.
