# `simulator/` — Forward simulators

> Wraps the synthetic portfolio generator as a forward model
> `simulate(theta) → x` for SBI training.

## Files

| File | Public API | Purpose |
|---|---|---|
| [`simulator/forward.py`](#forwardpy) | `ForwardSimulator`, `NumPyForwardSimulator`, `JAXForwardSimulator` | Forward model with NumPy and JAX backends |

The package exposes:

```python
from simulator import (
    ForwardSimulator, NumPyForwardSimulator, JAXForwardSimulator,
)
```

---

## `forward.py`

Abstract forward simulator `ForwardSimulator` plus two
concrete backends. The forward model takes a theta vector and
returns a 10-dimensional observation vector (the
`SyntheticPortfolioGenerator.losses_to_observation` summary).

### `ForwardSimulator(ABC)`

Abstract base. Subclasses must implement
`_simulate_batch(theta_batch, n_scenarios) → np.ndarray[B, 10]`.

Constructor:

```python
ForwardSimulator(portfolio_generator, regime="baseline", seed=42)
```

- `portfolio_generator`: a `SyntheticPortfolioGenerator`.
- `regime`: passed to the (unused-by-default) regime
  generator. Reserved for future integration.
- `seed`: random seed.

### Public methods

#### `simulate(theta_batch, n_scenarios=1000) → np.ndarray[B, 10]`

Vectorised batch simulation. Validates `theta_batch.ndim == 2`,
then dispatches to `_simulate_batch`. Returns `float32`.

#### `simulate_single(theta, n_scenarios=1000) → np.ndarray[10]`

Convenience wrapper: `simulate(theta[None, :])[0]`.

#### `grad_log_likelihood(theta, x_obs) → np.ndarray[D]`

Default implementation: central-difference finite differences
on `simulate_single`. Suitable for SBI methods that need a
gradient, though in practice SBI pipelines don't call this.

### `NumPyForwardSimulator`

Pure-NumPy backend. Loops over the batch sequentially:

```python
for b in range(B):
    dataset = self.portfolio_generator.sample(theta_batch[b], n_scenarios)
    obs = self.portfolio_generator.losses_to_observation(dataset.losses)
    observations[b] = obs
```

Suitable for CPU-only environments and small batches. For K=10
and `n_scenarios=1000`, the per-theta runtime is ~5 ms on a
single CPU core; the per-batch overhead is dominated by the
in-loop `_derive_loan_attributes` call (which is wasteful —
this could be moved out of the loop in a future optimisation).

### `JAXForwardSimulator`

JAX backend with JIT compilation. Tries to import `jax` /
`jax.numpy` / `scipy.stats.norm` at construction time. If any
import fails, `self._jax is None` and the simulator silently
falls back to `NumPyForwardSimulator` for every call.

Otherwise:

1. At first `simulate` call, builds a JIT-compiled batched
   simulation function (see `_build_jitted_simulate`).
2. Detects GPU via `jax.default_backend()`. If the platform
   is `cpu`, falls back to NumPy.
3. Runs the JIT'd function and converts back to NumPy.

#### `simulate_theta(theta, n_scenarios)` (JIT'd, inner)

The core JAX routine. Steps:

1. Sample systemic factor Z. If `tail_dep > 1e-6` and `nu < 100`,
   sample Z from a Student-t via `W / sqrt(V / nu)`. Otherwise
   Z is Gaussian. `n_scenarios` is a static argument (must be
   known at trace time for `jax.random.normal` to use concrete
   shapes).
2. Conditional default probabilities via probit link
   `(Φ⁻¹(p) - sqrt(ρ)·b·Z) / sqrt(1 - ρ·b²)`.
3. Bernoulli defaults via `jax.random.uniform(key, (K, n_scenarios))`.
4. Per-loan losses: `lgd · principal · default`.
5. Aggregate portfolio loss.
6. Compute 10-column observation vector (some columns are
   zero placeholders, see `data/synthetic.py`).

The whole function is wrapped in `@jit` and batched via `vmap`
over `theta`.

#### `_build_jitted_simulate()`

Constructs the JIT'd + vmap'd simulation function. Captures
`K`, `lgd_arr`, `principal_arr`, and `seed` from the parent.

#### `grad_log_likelihood(theta, x_obs)`

**Does not** use JAX autodiff. The comment in the source notes
that JAX random functions require concrete shapes, but the
gradient trace uses dynamic shapes — so the parent's
finite-difference implementation is used. This is a known
limitation and the recommended workflow is to use SBI
gradients internally rather than external
finite-differences.

### Performance comparison

| Backend | K=10, n=1000 | K=10, n=10^6 |
|---|---|---|
| NumPy | ~5 ms / theta | ~5 s / theta |
| JAX (CPU) | ~50 ms (incl. JIT) | ~3 s (incl. JIT) |
| JAX (GPU) | ~30 ms (incl. JIT, transfer) | ~0.5 s |

JAX is faster at scale on GPU, but NumPy is the more reliable
choice for CI / smoke tests.

### Usage example

```python
import numpy as np
from data.synthetic import SyntheticPortfolioGenerator
from simulator.forward import NumPyForwardSimulator

K = 10
gen = SyntheticPortfolioGenerator(K=K, seed=42)
sim = NumPyForwardSimulator(portfolio_generator=gen, seed=42)

theta = np.zeros(2 * K + 4, dtype=np.float32)
theta[:K] = 0.3
theta[K:2*K] = 0.02
theta[2*K + 1] = 0.5

x = sim.simulate_single(theta, n_scenarios=1000)
print(x.shape)  # (10,)
```
