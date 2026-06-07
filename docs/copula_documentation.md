# `copula/` — Factor-copula simulators

> The factor-copula layer. Each copula in this package turns a
> parameter vector `θ` and a number of scenarios into a
> `(U, losses)` pair, where `U[i, j] ~ Uniform(0, 1)` is the
> j-th loan's uniform marginal in scenario i and `losses[i]` is
> the aggregated portfolio loss.
>
> This is the layer that replaces `MultivariateGCI_Linear` for
> the posterior-propagated factor-copula workflow (architecture
> decision D2, D3).

## Common interface

All copulas expose:

```python
sample(theta: np.ndarray, n_samples: int = 1000)
  -> Tuple[np.ndarray, np.ndarray]   # (U: [n_samples, K], losses: [n_samples])
```

`theta` is a length-`(2K + 4)` vector laid out as:

```
theta[0:K]       = factor_loadings  b_i
theta[K:2K]      = p_zeros          p_i   (unconditional default probabilities)
theta[2K]        = tail_dep                (Gaussian: ignored)
theta[2K+1]      = rho              global correlation ρ
theta[2K+2]      = nu               dof (Student-t only)
theta[2K+3]      = spare
```

(For `LowRankFactorCopula` the layout is different; see its
section below.)

`losses_to_U_aggregated_loss(U, lgd, p_zeros)` is provided as a
**static method** on each copula, so you can recompute losses
without re-sampling.

## Files

| File | Class | Use case |
|---|---|---|
| [`copula/base.py`](#basepy) | `FactorCopula` (abstract) | Common interface |
| [`copula/gaussian.py`](#gaussianpy) | `GaussianFactorCopula` | One-factor Gaussian (default) |
| [`copula/student_t.py`](#studenttpy) | `StudentTFactorCopula` | One-factor Student-t (fat tails) |
| [`copula/vine.py`](#vinepy) | `DVineCopula` | Hand-rolled D-vine |
| [`copula/low_rank.py`](#lowrankpy) | `LowRankFactorCopula` | r-dimensional factor model |

---

## `base.py`

Abstract base class `FactorCopula(ABC)`. Subclasses must
implement:

- `_sample_impl(theta, n_samples) → (U, losses)` — the
  backend-specific sampling routine.
- `_validate_theta_shape(theta) → None` — validates shape and
  raises `ValueError` on mismatch.

The public `sample(theta, n_samples)` method calls
`_validate_theta` (which calls `_validate_theta_shape`) and then
`_sample_impl`. The static helper
`losses_to_U_aggregated_loss(U, lgd)` is **not used by the
Gaussian / Student-t / vine / low-rank subclasses** (they all
re-implement it as a static method on their own class). The
inherited default in `FactorCopula.losses_to_U_aggregated_loss`
contains a placeholder `defaults = (U < 0.5).astype(np.float32)`
and is **not** a correct implementation; use the per-copula
static methods instead.

---

## `gaussian.py`

**Class** `GaussianFactorCopula`. The **default** copula used by
`loader/posterior_factor_copula.py` and the rest of the project.

### Model

```
Z       ~ N(0, 1)                          (systemic factor)
ε_i     ~ N(0, 1)                          (idiosyncratic, i.i.d.)
X_i     = b_i · Z + sqrt(1 - b_i²) · ε_i   (latent variable for loan i)
default_i  iff  X_i < Φ⁻¹(p_i)
U_i     = Φ(X_i)                           (uniform marginal)
losses  = sum_i (default_i · lgd_i)
```

### Methods

- `sample(theta, n_samples) → (U, losses)`
  1. Clips `b` to `(-0.9999, 0.9999)` for numerical stability.
  2. Samples `Z` of shape `(n_samples,)` and `ε` of shape
     `(K, n_samples)`.
  3. Computes `X` as `b·Z + sqrt(1-b²)·ε` (shape `(K, n_samples)`).
  4. Defaults are `(X < Φ⁻¹(p))`, where `p` is clipped to
     `(1e-6, 1 - 1e-6)`.
  5. `U = Φ(X).T` (shape `(n_samples, K)`).
  6. `losses = defaults.T @ lgd` with `lgd = [0.40] * K`.

- `losses_to_U_aggregated_loss(U, lgd, p_zeros) → losses`
  Computes `losses = (U < p_zeros[None, :]) · lgd[None, :]` summed
  over columns. This is the public way to recompute losses
  **without re-sampling**.

### Performance

The implementation is fully vectorised with NumPy; for K=10 and
n_samples=1_000_000, sampling takes ~80 ms on a single CPU core.
For K=17 and n_samples=1_000_000, ~120 ms.

---

## `student_t.py`

**Class** `StudentTFactorCopula`. The same one-factor structure
as Gaussian, but with a **Student-t** systemic factor for
symmetric fat tails.

### Model

```
Z       = W / sqrt(V/ν)   with  W ~ N(0,1),  V ~ χ²(ν)
ε_i     ~ N(0, 1)
X_i     = b_i · Z + sqrt(1 - b_i²) · ε_i
default_i  iff  X_i < Φ⁻¹(p_i)
U_i     = Φ(X_i)
```

### Mapping `tail_dep → ν`

The `tail_dep` slot of `theta` is mapped to a degrees-of-freedom
value via:

```python
nu_effective = max(2.0, 100.0 - 98.0 · tail_dep)
```

- `tail_dep = 0` → `nu = 100` (near-Gaussian, no tail dep).
- `tail_dep = 1` → `nu = 2` (very fat tail, full tail dep).

This is the convention used in the paper; the t-copula's tail
dependence coefficient is then a function of `nu` and `rho`,
see `StudentTFactorCopula.tail_dependence_coefficient(rho, nu)`.

### `tail_dependence_coefficient(rho, nu) → float`

Static method. Computes the upper/lower tail-dependence
coefficient for a bivariate t-copula:

```
λ = 2 · (1 + ρ) / sqrt(1 - ρ) · T_{ν+2}(-sqrt((ν+1)(1-ρ)/(1+ρ)))
```

where `T_{ν+2}` is the standard t-distribution CDF with
`ν + 2` dof. For `ρ = 0` returns 0; for `|ρ| → 1` returns 1;
for `ν ≤ 2` returns 1.

### `losses_to_U_aggregated_loss(U, lgd, p_zeros) → losses`

Same as the Gaussian variant.

---

## `vine.py`

**Class** `DVineCopula`. Hand-rolled D-vine copula
(Bedford & Cooke 2002; Aas et al. 2009) with Gaussian pair
copulas.

### D-vine structure

For K variables ordered 1..K, the decomposition is:

```
f(x_1, ..., x_K) = f(x_1) · f(x_2|x_1) · f(x_3|x_1,x_2) · ... · f(x_K|x_1,...,x_{K-1})
```

The pair-copula construction (PCC) factorises each conditional
density into bivariate pair-copula densities. For a D-vine with K
variables, tree 1 has `K-1` pairs, tree 2 has `K-2`, etc.

### Sampling

This implementation samples sequentially:

```
U[:, 0] ~ Uniform(0, 1)
Z_1     = Φ⁻¹(U[:, 0])
for i in 1..K-1:
    Z_i | Z_{i-1} = z_prev  ~ N(ρ · z_prev, 1 - ρ²)
    U[:, i] = Φ(Z_i)
```

This is a **first-order Gaussian copula** along the vine; full
higher-order pair-copula terms are not implemented. For a full
D-vine, you'd need a dedicated library (e.g., `pyvine`).

### Limitations

- The full vine is combinatorial; this implementation works for
  `K ≤ 20` in practice.
- Higher-order pair-copula terms are approximated by the
  first-order Gaussian step.

### `losses_to_U_aggregated_loss(U, lgd, p_zeros) → losses`

Same pattern as the other copulas.

---

## `low_rank.py`

**Class** `LowRankFactorCopula`. Scalable path for K up to 1000
(architecture decision D2, scalability path).

### Model

```
Z       ~ N(0, I_r)                       (r-dim latent factor, r << K)
ε_i     ~ N(0, σ_eps²)
X_i     = Σ_{s=1..r} A_{i,s} · Z_s  +  ε_i   (K-vector equation)
X_std_i = (X_i - μ_i) / σ_i               (column-wise z-score)
default_i  iff  X_std_i < Φ⁻¹(p_i)
U_i     = Φ(X_std_i)
```

This is a **rank-r factor model**: the covariance matrix of `X`
is `A·A.T + σ_eps²·I`, which is rank-r + diagonal. The advantage
over a full K×K covariance is `O(r·K)` storage and `O(r·K·n)`
sampling cost.

### Theta layout

```
theta[0 : K·r]         = A matrix (K×r, row-major flatten)
theta[K·r : K·r + K]   = p_zeros
theta[K·r + K]         = tail_dep (ignored)
theta[K·r + K + 1]     = σ_eps (idiosyncratic std; default 1.0)
theta[K·r + K + 2..3]  = spare
```

Total length: `K·r + K + 4`.

### Standardisation

For `U = Φ(X)` to be `Uniform(0, 1)`, the columns of `X` are
z-scored **after** the linear combination. This is a
common approximation; the alternative is to standardise the
theoretical moments of `A·A.T + σ_eps²·I` analytically, but the
empirical z-score converges quickly for `n_samples ≫ K`.

### `losses_to_U_aggregated_loss(U, lgd, p_zeros) → losses`

Same pattern as the other copulas.

---

## Usage example

```python
import numpy as np
from copula.gaussian import GaussianFactorCopula

K = 10
theta = np.zeros(2 * K + 4, dtype=np.float32)
theta[:K] = 0.3                  # factor loadings
theta[K:2*K] = 0.02              # default probs
theta[2*K + 1] = 0.5             # correlation

copula = GaussianFactorCopula(K=K, seed=42)
U, losses = copula.sample(theta, n_samples=100_000)

# Or recompute losses without re-sampling:
losses_again = GaussianFactorCopula.losses_to_U_aggregated_loss(
    U, lgd=np.full(K, 0.40, dtype=np.float32), p_zeros=theta[K:2*K]
)
```

See the notebooks in the project root and the tests under
[`tests/test_copula_*.py`](tests_documentation.md) for more
usage examples.
