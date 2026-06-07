# `sbi_pipeline/` — Simulation-Based Inference

> Three SBI training pipelines (NPE / NLE / flow-matching)
> built on `sbi 0.22.0`. The primary inference engine for the
> posterior over factor-copula parameters.

## Files

| File | Public API | Purpose |
|---|---|---|
| [`sbi_pipeline/posterior.py`](#posteriorpy) | `NPETrainingPipeline`, `NLETrainingPipeline`, `FlowMatchingTrainingPipeline`, `SBIPosterior` | The three SBI pipelines |
| [`sbi_pipeline/utils.py`](#utilspy) | `SBITrainingConfig`, `get_prior_from_bounds`, `WandbLoggingHook`, `NoOpLoggingHook`, `get_logging_hook` | Configuration and W&B helpers |

The package exposes:

```python
from sbi_pipeline import (
    NPETrainingPipeline, NLETrainingPipeline, FlowMatchingTrainingPipeline,
    SBIPosterior, SBITrainingConfig, get_prior_from_bounds,
)
```

### Memory note (from prior research)

> sbi 0.22.0: `from sbi.utils import posterior_nn`
> `flow._prior` → `flow.prior` (deprecated API fix)
> `CpuPrior` wrapper for `.log_prob()` and `.sample()` compatibility
> `n_workers=1` for JAX thread safety

These are the gotchas that the research run's pipelines
explicitly work around.

---

## `posterior.py`

The core SBI layer. ~1650 lines. Most of the file is
boilerplate (masked linear layers, MADE blocks, cMAF
coupling) that could be replaced by `nflows` or `pyro` in
production, but the project ships a self-contained
implementation.

### Top-level container

```python
@dataclass
class TrainingResult:
    posterior: "SBIPosteriorWrapper"
    log_probs: list[float]   # mean log-prob per round
    ess_values: list[float]  # effective sample size estimate per round
    wandb_url: str | None
```

### `CpuPrior(dist.Distribution)`

CPU-compatible prior wrapper. Stores `low` and `high` as
`torch.Tensor`s and delegates to `torch.distributions.Uniform`.
The wrapper exists because `sbi.utils.BoxUniform` has
device-placement issues on some configurations; the wrapper
forces everything to CPU.

### Conditional Masked Autoregressive Flow

The cMAF is the flow-matching pipeline's density estimator.
Four classes:

1. `MaskedLinear(in_features, out_features, mask)` — applies a
   binary mask to the linear weights before the linear
   transform. Used to enforce the MADE autoregressive
   property.
2. `MADEBlock(n_in, n_hidden, n_out, mask_seed=42)` — a stack
   of `MaskedLinear` layers with lower-triangular masks. Each
   output dimension depends only on input dimensions `< output_index`.
3. `ConditionalMAF(context_dim, theta_dim, n_layers=4, hidden_dims=None)` —
   the full flow. Stacks `n_layers` `_CouplingLayer` instances
   that each transform half the dimensions via
   `(θ_even, θ_odd) ↦ (θ_even · scale + shift, θ_odd)`, where
   `scale` and `shift` are MADEBlock outputs conditioned on
   `(θ_odd, x)`.
4. `_CouplingLayer(theta_dim, context_dim, hidden_dim, layer_idx)` —
   the single-layer building block. Uses two `MADEBlock`s
   (one for `scale`, one for `shift`).

`ConditionalMAF` exposes:

- `forward(theta, x) → (z, log_det)` — change-of-variables.
- `inverse(z, x) → theta` — ancestral sampling.
- `log_prob(theta, x) → log_prob` — `log p(θ|x)`.
- `sample(n_samples, x) → samples` — `n_samples` from `p(θ|x)`.

### `_CouplingLayer`

The standard RealNVP-style coupling:

```
θ_even, θ_odd = θ[:, :split_dim], θ[:, split_dim:]
cond = cat([θ_odd, x], dim=-1)
scale = exp(scale_net(cond))   # MADEBlock
shift = shift_net(cond)        # MADEBlock
θ_even_out = θ_even * scale + shift
```

Log-det-Jacobian contribution: `sum(log(scale), dim=-1)`.

### Wrappers

#### `CMAFWrapper(cmaf, prior, context_dim)`

Wraps `ConditionalMAF` with `sbi`-style `.sample()` /
`.log_prob()` methods. Stores a "default x" context that
must be set via `set_default_x(x)` before sampling.

#### `_CMAFCompatiblePosterior(cmaf_wrapper, prior, method="flow_matching")`

Adapter that gives the cMAF a `sbi`-compatible
`._neural_net` attribute (for the
`SBIPosteriorWrapper._neural_net` property).

#### `SBIPosteriorWrapper(posterior, prior, method="npe" | "nle" | "flow_matching")`

Unified wrapper exposing a consistent interface across all
three pipelines:

- `sample(sample_shape) → np.ndarray` — draw from
  `p(θ|x)`. Auto-detects whether `sample_shape` is an int
  or a tuple.
- `log_prob(theta) → np.ndarray` — `log p(θ|x)`.
- `coverage_check_marginal(test_thetas, test_xs, alpha_levels=[0.05, 0.5, 0.95], tolerance=0.05) → dict`
  — Simulation-Based Calibration (SBC) coverage check on
  marginal ranks. Returns:
  ```python
  {
      "ranks": np.ndarray[N],            # per-test-point average rank
      "coverage_errors": {              # per-alpha empirical minus nominal
          "alpha_0.05":  float, ...
      },
      "passed": bool,                    # all errors < tolerance
  }
  ```
- `cdf(theta, x) → np.ndarray` — **not implemented**; raises
  `NotImplementedError` ("use rank-based SBC instead").
- `coverage_check(...)` — deprecated alias for
  `coverage_check_marginal`.

The `SBIPosteriorWrapper.__init__` patches a deprecated-API
issue with sbi 0.22.0: if the underlying posterior has a
`_prior` attribute, it sets `posterior.prior = posterior._prior`
so the newer accessor works.

### `NPETrainingPipeline`

Neural Posterior Estimation (primary method). Uses sbi's
`SNPE(prior, density_estimator=neural_net)` with
`posterior_nn(model="maf", hidden_features=50, num_transforms=4)`.

The `train` method wraps sbi's `inference.append_simulations`
→ `flow.train` → `inference.build_posterior` loop:

- For round 0, uses all `training_pairs`.
- For round > 0, samples from the current proposal
  (`posterior` if `(round_idx + 1) % 5 == 0`, else prior).
- After training, builds the posterior with
  `inference.build_posterior(flow._neural_net)` and calls
  `posterior.set_default_x(x_tensor[0])` (sbi 0.22.0 quirk).

`train_from_simulator(simulator, n_initial=500, ...)` is a
convenience wrapper that auto-generates `(θ, x)` pairs by
sampling θ from the prior and running the simulator.

The method is auto-throttled: if CUDA is requested but not
available, falls back to CPU silently.

### `NLETrainingPipeline`

Neural Likelihood Estimation. Uses sbi's
`SNLE(prior, density_estimator=neural_net)` with
`posterior_nn(model="nsf", hidden_features=50, num_transforms=4)`.

Same overall training loop as NPE, but with neural spline
flows instead of MAF. The `train_from_simulator` wrapper is
identical.

### `FlowMatchingTrainingPipeline`

Conditional MAF training. Does **not** use sbi at all — it
trains the `ConditionalMAF` directly via maximum likelihood
on `(θ, x)` pairs, then wraps it in `CMAFWrapper` and
`_CMAFCompatiblePosterior`.

The training loop:

1. Build a `ConditionalMAF(context_dim=10, theta_dim=D, n_layers=4, hidden_dims=[hidden_features, hidden_features])`.
2. Optimise with Adam (`lr=5e-4`).
3. Per round, shuffle the data and iterate over minibatches
   of size `batch_size`. Each minibatch: `loss = -cmaf.log_prob(theta, x).mean()`.
4. Diagnostics: `eval_log_prob` on the first 10 θ vectors.

Returns a `TrainingResult` whose `posterior` is a
`_CMAFCompatiblePosterior` (which has the same `.sample()` /
`.log_prob()` interface as the NPE/NLE posteriors).

### `SBIPosterior`

A thin factory that wraps one of the three pipelines:

```python
SBIPosterior(
    prior, method="npe"|"nle"|"flow_matching",
    hidden_features=50, device="cuda", seed=42, wandb_project=None,
)
```

- `train(training_pairs, ...) → TrainingResult`
- `sample(posterior, sample_shape) → np.ndarray`
- `log_prob(posterior, theta) → np.ndarray`
- `coverage_check(posterior, test_thetas, test_xs, alpha_levels=None) → dict`

The factory exists to give the SBI layer a single
configuration entry point (so the same `SBITrainingConfig`
works for all three methods).

### Caveats

- The NPE/NLE pipelines wrap sbi 0.22.0 and inherit its
  quirks (CPU prior wrapping, `flow._neural_net` access, etc.).
- The flow-matching pipeline is a **self-contained**
  implementation; it does **not** use sbi at all. The
  cMAF architecture is structurally distinct from sbi's
  MAF, so the two are not interchangeable.
- The `coverage_check_marginal` implementation computes
  **average-of-marginal-ranks**, not the multivariate
  CDF probability `P(θ' < θ_i)`. For correlated parameters,
  use a multivariate rank test instead.
- ESS (effective sample size) is approximated as
  `min(200, len(training_pairs) / (round_idx + 1))` — a
  very rough placeholder. Replace with proper ESS from
  the trained posterior in production.

---

## `utils.py`

Configuration dataclass + W&B hooks + prior helpers.

### `SBITrainingConfig`

```python
@dataclass
class SBITrainingConfig:
    prior: dist.Distribution
    n_rounds: int = 10
    n_simulations_per_round: int = 1000
    batch_size: int = 100
    learning_rate: float = 5e-4
    hidden_features: int = 50
    num_transforms: int = 4
    device: str = "cuda"
    wandb_project: Optional[str] = None
    seed: int = 42
```

`__post_init__` validates each field and raises
`ValueError` on invalid input. Used by
`experiments/sbi_train.py` for CLI argument parsing.

### `get_prior_from_bounds(low, high) → BoxUniform`

Builds a `sbi.utils.BoxUniform` from NumPy bounds. Validates
that `low.shape == high.shape` and that all `low < high`.

### `WandbLoggingHook`

W&B logging hook used as a context manager. Initialises a
`wandb.init(project, name, config)` run on `__enter__` and
calls `wandb.run.finish()` on `__exit__`. The `log(metrics,
step=None)` and `log_summary(summary)` methods forward to
`wandb.log` / `wandb.run.summary`. If `wandb` is not
imported, the hook is a no-op (degrades gracefully).

### `NoOpLoggingHook`

Stub for the no-W&B case. Same API as `WandbLoggingHook`
but every method is a no-op.

### `get_logging_hook(project, name=None, config=None) → WandbLoggingHook | NoOpLoggingHook`

Factory. Returns a `WandbLoggingHook` if `wandb` is
imported and `project is not None`, else a `NoOpLoggingHook`.
