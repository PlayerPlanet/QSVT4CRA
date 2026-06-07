# `metrics/` — Classical risk metrics & quantum-vs-classical error

> The metrics layer. Provides the classical benchmark
> (`var_cvar`, `GroundTruthMC`) and the quantum-vs-classical
> error analysis (`quantum_error`).

## Files

| File | Public API | Purpose |
|---|---|---|
| [`metrics/var_cvar.py`](#varcvarpy) | `var_at`, `cvar_at`, `var_cvar`, `loss_cdf` | VaR / CVaR / empirical CDF |
| [`metrics/ground_truth.py`](#groundtruthpy) | `GroundTruthMC` | Massive-MC ground truth engine |
| [`metrics/quantum_error.py`](#quantumerrorpy) | `quantum_vs_classical_error`, `cdf_error`, `tail_error`, `var_error`, `cvar_error` | Quantum circuit error analysis |

The package exposes:

```python
from metrics import (
    var_cvar, var_at, cvar_at, loss_cdf,
    GroundTruthMC,
)
```

---

## `var_cvar.py`

Pure-NumPy risk metrics. The benchmark that the quantum
circuit's output is compared against.

### `loss_cdf(losses, x_grid) → np.ndarray[M]`

Empirical CDF of `losses` evaluated at the points in `x_grid`:

```python
cdf = np.sum(losses[None, :] <= x_grid[:, None], axis=1) / N
```

Returns a `float32` array of shape `(M,)` where `M = len(x_grid)`.

### `var_at(losses, alpha) → float`

Value-at-Risk at confidence level `alpha`:

```python
var = np.quantile(losses, alpha)
```

Raises `ValueError` if `losses` is empty or `alpha ∉ (0, 1)`.
Uses NumPy's default linear interpolation.

### `cvar_at(losses, alpha) → float`

Conditional Value-at-Risk (Expected Shortfall) at confidence
level `alpha`:

```python
var = np.quantile(losses, alpha)
tail = losses[losses > var]
cvar = tail.mean() if tail.size > 0 else var
```

Strictly `cvar ≥ var` for any sample.

### `var_cvar(losses, alphas=[0.95, 0.99, 0.999]) → dict`

Convenience wrapper that computes VaR/CVaR at multiple
confidence levels in one call. Returns:

```python
{
    "var_0_95":  float, "cvar_0_95":  float, "tail_prob_0_95":  float,
    "var_0_99":  float, "cvar_0_99":  float, "tail_prob_0_99":  float,
    "var_0_999": float, "cvar_0_999": float, "tail_prob_0_999": float,
}
```

`tail_prob_alpha = mean(losses > var_alpha)`, which empirically
should be close to `1 - alpha`. The key naming uses
`str(alpha).replace(".", "_")` so the dict keys are
file-name-safe.

### Doctests

The module includes doctests in `loss_cdf`, `var_at`, and
`var_cvar` docstrings; running `pytest --doctest-modules
metrics/var_cvar.py` validates them.

---

## `ground_truth.py`

**Class** `GroundTruthMC(copula, portfolio_generator, n_scenarios=1_000_000, posterior_samples=None, regime="baseline", seed=42)`.

Massive-Monte-Carlo ground-truth engine for the
**posterior-predictive** VaR/CVaR distribution. This is the
reference benchmark that `metrics.quantum_error` compares
quantum circuit outputs to.

### `ALPHAS = [0.95, 0.99, 0.999]`

Default confidence levels used throughout the project.

### Constructor

- `copula`: a `FactorCopula` instance (Gaussian, Student-t).
- `portfolio_generator`: a `SyntheticPortfolioGenerator`
  providing the LGD / principal / region structure.
- `n_scenarios`: base number of MC scenarios per posterior
  sample (default 1e6).
- `posterior_samples`: array of shape `(N, D)`. If `None`, a
  default theta is generated via `_default_theta()`.
- `regime`: stress regime name (for logging/labeling only).
- `seed`: random seed.

### `_default_theta() → np.ndarray[1, D]`

Generates a single default posterior sample when none is
provided. Random uniform factor loadings, p_zeros, etc.
Used for the test suite and quick smoke tests.

### `run(samples_per_posterior=None, store_all_losses=False) → dict`

The standard MC loop. For each posterior sample θ⁽ⁱ⁾:

1. `_, losses_i = copula.sample(theta_i, n_samples=samples_per_posterior)`.
2. `var_95[i] = np.quantile(losses_i, 0.95)`.
3. `tail_95 = losses_i[losses_i > var_95[i]]`;
   `cvar_95[i] = tail_95.mean() if tail_95.size > 0 else var_95[i]`.
4. (same for 99, 99.9).

Aggregates into posterior-predictive summaries
(`predictive_var_at_*` = mean across posterior samples) and
returns a dict with per-sample + summary metrics.

The `store_all_losses=True` flag concatenates all loss samples
into a single array and returns it; this is **memory-heavy**
(e.g., 1e7 × 4 bytes = 40 MB per posterior sample) and
disabled by default with a guard `total_samples < 1e8`.

### `run_streaming(samples_per_posterior, batch_size=50_000) → dict`

Memory-efficient variant. Streams losses in batches of
`batch_size` and concatenates **only the current posterior's
losses** (not all posteriors' losses), so peak memory is
`O(batch_size)` rather than `O(N · samples_per_posterior)`.
Use this when `samples_per_posterior ≥ 1e7`.

The two methods return the same dict structure:

| Key | Type | Description |
|---|---|---|
| `posterior_var`, `posterior_cvar` | `(N,)` | VaR/CVaR at 95% per sample |
| `posterior_var_99`, `posterior_cvar_99` | `(N,)` | at 99% |
| `posterior_var_999`, `posterior_cvar_999` | `(N,)` | at 99.9% |
| `predictive_var_at_*` | `float` | posterior-mean VaR at each α |
| `predictive_cvar_at_*` | `float` | posterior-mean CVaR at each α |
| `all_loss_samples` | `np.ndarray` or `None` | only if `store_all_losses=True` |
| `n_posterior_samples` | `int` | |
| `n_scenarios_per_posterior` | `int` | |
| `regime` | `str` | |
| `runtime_seconds` | `float` | |

### Doctests in module docstring

The class has a runnable doctest in its class docstring;
running `pytest --doctest-modules metrics/ground_truth.py`
validates the end-to-end run.

---

## `quantum_error.py`

Functions to compare quantum circuit measurement results to
classical ground truth.

### `quantum_vs_classical_error(qc, classical_value, n_shots=10_000, backend="aer_simulator") → dict`

Runs the quantum circuit `qc` on a Qiskit Aer simulator for
`n_shots`, extracts the marginal probability of the objective
qubit being `|1⟩`, and compares to `classical_value`.

**Objective qubit convention**: the last qubit of the circuit
is treated as the answer bit. Counts are parsed assuming
Qiskit's little-endian bitstring convention (the last character
of the string is the LSB).

Returns:

```python
{
    "quantum_estimate": float,
    "classical_value":  float,
    "abs_error":        float,
    "rel_error":        float,
    "ci_95":            (float, float),  # 95% normal-approx CI on the estimate
}
```

If `qiskit_aer` is not installed or the simulator run fails,
returns the classical value with zero error (the function is
designed to fail gracefully in CI environments).

### `cdf_error(qc, x_grid, classical_cdf, n_shots=10_000) → dict`

For each point in `x_grid`, runs the quantum circuit (which
encodes the indicator `1{loss ≤ x}`) and computes the
Kolmogorov-Smirnov statistic against the classical CDF. Returns:

```python
{
    "ks_statistic": float,
    "quantum_cdf":  np.ndarray,  # same length as x_grid
    "classical_cdf": np.ndarray,
}
```

### `tail_error(qc, target, classical_tail_prob, n_shots=10_000) → dict`

Specialised comparison for the tail probability
`P(loss > target)` that the QSVT risk circuit estimates.
Returns:

```python
{
    "quantum_tail_prob":    float,
    "classical_tail_prob":  float,
    "abs_error":            float,
    "rel_error":            float,
}
```

### `var_error(qc, alphas, classical_var, n_shots=10_000) → dict`

Per-alpha quantum-vs-classical VaR error. `classical_var` is
expected to be a dict keyed by `str(alpha).replace(".", "_")`,
e.g. `{"0_95": 0.5, "0_99": 0.7}`.

### `cvar_error(qc, alphas, classical_cvar, n_shots=10_000) → dict`

Same as `var_error` for CVaR.

### Caveats

- All functions assume the **last qubit** of the circuit is
  the answer bit. If the circuit measures a different qubit,
  wrap it in a custom permutation or transpile.
- The 95% CI uses the normal approximation
  `p ± 1.96 · sqrt(p(1-p)/n_shots)`. For very small/large
  `p` (close to 0 or 1) this is conservative.
- The functions are designed to fail gracefully when
  `qiskit_aer` is missing — the CI environment does not
  install it, and the metrics should still be importable
  without it.
