# `tests/` — pytest suite

> The test suite. ~25 test modules covering every component
> of the research run. All tests are runnable as
> `pytest tests/`.

## Files

| Test file | Module under test | Notes |
|---|---|---|
| [`tests/conftest.py`](#conftestpy) | (fixtures) | Shared fixtures |
| [`tests/test_code_qsvt_equivalence.py`](#test_code_qsvt_equivalencepy) | `Code/QSVT.py` | Statevector equivalence tests |
| [`tests/test_compiler_backend_heron.py`](#test_compiler_backend_heronpy) | `compiler_backend/heron.py` | Pure-helper unit tests |
| [`tests/test_compiler_backend_token.py`](#test_compiler_backend_tokenpy) | `compiler_backend/heron.py` | Token-resolution tests |
| [`tests/test_copula_gaussian.py`](#test_copula_gaussianpy) | `copula/gaussian.py` | |
| [`tests/test_copula_integration.py`](#test_copula_integrationpy) | All copulas | Integration tests |
| [`tests/test_copula_low_rank.py`](#test_copula_low_rankpy) | `copula/low_rank.py` | |
| [`tests/test_copula_student_t.py`](#test_copula_student_tpy) | `copula/student_t.py` | |
| [`tests/test_copula_vine.py`](#test_copula_vinepy) | `copula/vine.py` | |
| [`tests/test_data_stress_regimes.py`](#test_data_stress_regimespy) | `data/stress_regimes.py` | |
| [`tests/test_data_synthetic.py`](#test_data_syntheticpy) | `data/synthetic.py` | |
| [`tests/test_experiments_qsvt_sweep.py`](#test_experiments_qsvt_sweeppy) | `experiments/qsvt_sweep.py` | |
| [`tests/test_forward_simulator.py`](#test_forward_simulatorpy) | `simulator/forward.py` | |
| [`tests/test_loader_amplitude.py`](#test_loader_amplitudepy) | `loader/amplitude_loader.py` | |
| [`tests/test_loader_posterior_factor_copula.py`](#test_loader_posterior_factor_copulapy) | `loader/posterior_factor_copula.py` | |
| [`tests/test_lumi_scaling.py`](#test_lumi_scalingpy) | `scale_runner.py` + dispatcher | LUMI dispatching tests |
| [`tests/test_make_figures.py`](#test_make_figurespy) | `experiments/make_figures.py` | |
| [`tests/test_metrics_quantum_error.py`](#test_metrics_quantum_errorpy) | `metrics/quantum_error.py` | |
| [`tests/test_metrics_var_cvar.py`](#test_metrics_var_cvarpy) | `metrics/var_cvar.py`, `metrics/ground_truth.py` | |
| [`tests/test_ood_robustness.py`](#test_ood_robustnesspy) | `experiments/ood_robustness.py` | |
| [`tests/test_phase1_sbc.py`](#test_phase1_sbcpy) | Phase 1 SBI gate | Publication-grade SBC validation |
| [`tests/test_qsvt_approximator.py`](#test_qsvt_approximatorpy) | `qsvt/approximator.py` | |
| [`tests/test_qsvt_circuit.py`](#test_qsvt_circuitpy) | `qsvt/circuit.py` | |
| [`tests/test_qsvt_threshold.py`](#test_qsvt_thresholdpy) | `qsvt/threshold.py` | |
| [`tests/test_resource_scaling.py`](#test_resource_scalingpy) | `experiments/resource_scaling.py` | |

`pytest.ini` (project root) configures the test runner. The
default is `pytest tests/`.

---

## `conftest.py`

Shared fixtures:

- `K = 10` — default portfolio size.
- `theta_baseline(K)` — a fresh `np.ndarray` of shape
  `(2K+4,)` with random factor loadings, p_zeros, and
  default copula params. Each call returns a new array
  (the `.copy()` at the end is important to avoid mutation
  across tests).
- `regime_gen` — `data.stress_regimes.StressRegimeGenerator(seed=42)`,
  gated by `pytest.importorskip`.

---

## `test_code_qsvt_equivalence.py`

**The most important test in the suite.** These are
deterministic statevector equivalence tests for the
`Code/QSVT.py` construction. They protect against math
regressions when the `addProj` / `createQSVT` / boundary
projector rewrites go in.

Approach:

1. Build the same logical QSVT (same U, same phases) with
   the **current** implementation.
2. Capture the statevector via
   `qiskit.quantum_info.Statevector.from_instruction`.
3. The refactored version (or the same version on a future
   commit) must match within `1e-9` on all `2^n` amplitudes.

The reference sub-circuit (in
`_build_sub_circuit`) is the same as
`Code/circuitsCRA.get_expected_probability_circuit`, so the
test exercises the same sub-circuit the production pipeline
uses.

`small_qsvt_inputs` fixture: K=4 (so the statevector is
`2^6 = 64` amplitudes, fast to compute), degree=4
(5 phases), random LGDs, random phases. The test asserts
that the statevector equals a saved "golden" reference with
absolute tolerance `1e-9`.

When the QSVT math changes, the golden statevector
**must** be regenerated and re-committed; the test framework
cannot distinguish a "math change that's correct" from a
"math change that's a regression" — it just compares to the
saved value.

---

## `test_compiler_backend_heron.py`

Pure-helper unit tests for `compiler_backend/heron.py`. The
test file uses `qiskit.QuantumCircuit` to build small test
backends and exercises:

- `add_measurements` (all-qubit and selective).
- `_backend_name`, `_processor_type`, `_target`,
  `_num_backend_qubits` with various backend-like mocks.
- `_iter_coupling_edges` (with target, with `coupling_map`,
  with `configuration().coupling_map`).
- `_instruction_error_from_target`, `_gate_error`,
  `_readout_error`, `_edge_weight`.
- `select_calibration_aware_layout` with a small mock
  backend.
- `_two_qubit_count` (counts CX, CZ, ECR, etc.).
- `_success_probability_proxy` (returns a float in `(0, 1]`
  for non-empty inputs).
- `_try_pyzx_optimize` (returns `(circuit, error_msg)` if
  pyzx is missing; `(optimized, None)` on success).
- `compile_for_backend` end-to-end with a mock backend.

All tests use mocks (no real IBM token needed).

---

## `test_compiler_backend_token.py`

Token-resolution tests. The IBM token must be resolved
from, in order:

1. `$IBM_API_KEY` (default; recommended for HPC / Slurm).
2. `$IBM_API_KEY_FILE` (path to a file whose first non-empty,
   non-comment line is the token).
3. `token_file` (default `.ibm_token` in cwd).

The tests cover:

- `load_ibm_token()` raises `RuntimeError` with a clear
  message when none of the three are available.
- `load_ibm_token()` returns the value of `$IBM_API_KEY`
  when it is set.
- `load_ibm_token()` reads the file at `$IBM_API_KEY_FILE`
  when set.
- `load_ibm_token()` reads `.ibm_token` when neither env
  var is set.
- The token is **never** logged (asserted via
  `capsys.readouterr()`).
- Comment lines (starting with `#`) in the token file are
  skipped.
- Blank lines are skipped.
- `token_file=None` disables the file fallback entirely.

---

## `test_copula_*.py`

Per-copula unit tests. Each file covers:

- `sample(theta, n)` returns arrays of the right shape and
  dtype (`float32`).
- `sample` validates `theta.shape == (2K + 4,)`.
- The seed produces reproducible output.
- The static `losses_to_U_aggregated_loss` is consistent
  with `sample`.
- For Gaussian: `U` is approximately `Uniform(0, 1)` (KS
  test, p > 0.01).
- For Student-t: tail dependence coefficient increases as
  `nu` decreases; equals 0 at `rho=0`.
- For D-vine: the U marginals are uniform; copula structure
  induces correlation.
- For low-rank: the rank-r structure produces a covariance
  matrix with the expected `r + K` non-zero eigenvalues.

### `test_copula_integration.py`

Integration tests that wire each copula into a
`SyntheticPortfolioGenerator` and check the end-to-end
`PortfolioDataset` shape and content. Validates that the
copula loss array matches what the generator would produce
via its own `_simulate_losses`.

---

## `test_data_*.py`

#### `test_data_synthetic.py`

Tests `data/synthetic.py`:

- `SyntheticPortfolioGenerator.sample(theta, n_scenarios)`
  returns a `PortfolioDataset` with the right shape and
  dtype.
- The `observations` matrix has 10 columns.
- `theta` is preserved exactly.
- `theta.shape != (2K+4,)` raises `ValueError`.
- The 10 observation columns are computed correctly (e.g.,
  `var95 = np.percentile(losses, 95)`).
- `_derive_loan_attributes` produces LGDs in `[0.10, 0.80]`,
  principals in `[€50k, €500k]`, region indices in
  `{0, 1, 2, 3, 4}`.
- `_simulate_losses` is reproducible given the same seed.

#### `test_data_stress_regimes.py`

Tests `data/stress_regimes.py`:

- `StressRegimeGenerator.sample(regime, theta, shock)`
  raises `ValueError` for unknown regimes.
- `baseline` is the identity.
- Each stress regime modifies the right slots of `theta`:
  - `housing_crash` increases `p_zeros`, `tail_dep`, and
    `rho`.
  - `rate_shock` increases `p_zeros` more aggressively.
  - `unemployment` increases `p_zeros` for the bottom half
    of factor loadings.
  - `liquidity` decreases `nu` and increases `tail_dep`.
- Shock `0` is the identity.
- The output is `float32`.

---

## `test_experiments_qsvt_sweep.py`

Tests `experiments/qsvt_sweep.py`:

- `run_sweep(degrees, posterior_samples, K, target_loss, n_shots)`
  returns a dict with `per_degree` keyed by degree.
- The output `.npz` has the expected keys (`d{deg}_*`).
- Failures at a given degree are captured (the entry has
  `error` key, not the metric keys).
- The default posterior sample (when `posterior_samples=None`)
  is `np.zeros(2K+4)` plus a default tail-dep slot.

---

## `test_forward_simulator.py`

Tests `simulator/forward.py`:

- `NumPyForwardSimulator.simulate(theta, n_scenarios)`
  returns a `float32` array of shape `(10,)`.
- `simulate(theta_batch, n_scenarios)` returns
  shape `(B, 10)`.
- `simulate_single(theta, n_scenarios)` is equivalent to
  `simulate(theta[None, :])[0]`.
- The result is reproducible given the same seed.
- The JAX backend falls back to NumPy when JAX is not
  installed.
- The NumPy backend gives the same result as the JAX backend
  when JAX is available (modulo RNG differences).
- `grad_log_likelihood` returns a vector of shape `(D,)`.

---

## `test_loader_*.py`

#### `test_loader_amplitude.py`

Tests `loader/amplitude_loader.py`:

- `AmplitudeLoader(K, values)` validates `len(values) == 2^K`.
- The circuit has `K + 1` qubits.
- `num_ancillas == 0`.
- `to_gate()` returns a `Gate` object.
- The rotation angles are correctly derived from the values
  (verified against an analytical expression).

#### `test_loader_posterior_factor_copula.py`

Tests `loader/posterior_factor_copula.py`:

- `PosteriorFactorCopulaLoader(theta, K, max_loss)` builds a
  circuit with `K + 1` qubits.
- `K` and `lgd` properties return the right values.
- The internal `_compute_scenario_probs` produces a
  normalised histogram over `2^K` bins.
- The circuit decomposes without error.
- For `K=4` the statevector amplitude on the all-zeros
  state is the `sqrt(P(scenario=0))` from the copula, to
  within 1e-6 (compared to a manual `np.bincount`).

---

## `test_lumi_scaling.py`

Tests `scale_runner.py` and the LUMI dispatching logic:

- `detect_environment()` returns a dict with the expected
  keys (`nnodes`, `ntasks`, `ntasks_per_node`, etc.).
- `get_launcher()` returns the right concrete class for each
  detected environment (`sequential`, `joblib_local`,
  `joblib_srun`, `torch_ddp`).
- Each launcher's `map` works on a small test workload.
- The dispatcher (`lumi_deployment/dispatcher.sh`) sourceable
  without error and exposes the `EXPERIMENT` case statement
  (validated by grepping for known experiment names).
- The `slurm_*.sh` scripts source `setup_lumi_env.sh` and
  the right Python entry point.

---

## `test_make_figures.py`

Tests `experiments/make_figures.py`:

- `FigureGenerator(results_dir, output_dir, K, seed)` builds
  the synthetic fallback data without error.
- `figureN_*` methods each return a `Path` to the saved PNG.
- `run_all()` returns a dict mapping figure name to path.
- All 8 figures are produced (or attempted; failure of one
  doesn't abort the rest).
- Figures are saved at the requested DPI.

---

## `test_metrics_*.py`

#### `test_metrics_quantum_error.py`

Tests `metrics/quantum_error.py`:

- `quantum_vs_classical_error` returns a dict with the
  expected keys.
- The function degrades gracefully when `qiskit_aer` is not
  installed (returns zero error).
- `cdf_error` computes a non-negative `ks_statistic`.
- `tail_error`, `var_error`, `cvar_error` return dicts with
  the right structure.

#### `test_metrics_var_cvar.py`

Tests `metrics/var_cvar.py` and `metrics/ground_truth.py`:

- `loss_cdf(losses, x_grid)` returns the right shape and
  monotonicity.
- `var_at` raises `ValueError` on empty `losses` or
  `alpha ∉ (0, 1)`.
- `var_at` is monotonic in `alpha`.
- `cvar_at ≥ var_at` for any sample.
- `var_cvar` returns the expected dict.
- `GroundTruthMC.run(n_scenarios=1000, posterior_samples=...)`
  returns a dict with the right structure.
- `GroundTruthMC.run_streaming(...)` returns the same
  structure (modulo `all_loss_samples=None`).

The tests also exercise the module-level doctests via
`pytest --doctest-modules`.

---

## `test_ood_robustness.py`

Tests `experiments/ood_robustness.py`:

- `OODExperiment(posterior_samples, test_regimes, n_scenarios, K, copula, seed)` builds without error.
- `.run()` returns a dict with `regime_results` keyed by
  regime.
- Each `regime_results[regime]` has both `method_A_gci` and
  `method_B_posterior` keys.
- Method B's per-sample VaR/CVaR arrays have the right
  shape.
- `_resolve_shock` maps `rate_shock_0.5 → (rate_shock, 0.5)`
  and `rate_shock_1.5 → (rate_shock, 1.5)`.
- `compare_methods_plot(results, output_path)` produces a
  PNG at the given path.

---

## `test_phase1_sbc.py`

**The publication-grade Phase 1 ML gate.** A long
test module (~50 KB) that implements the
Simulation-Based Calibration validation gate that blocks
the project from advancing to Phase 2 if the SBI posterior
fails the coverage check.

Substantial SBC test machinery. Refer to the file directly
for details; the high-level flow is:

1. Train the SBI posterior on `(θ, x)` pairs sampled from
   the prior + forward simulator.
2. Sample `θ⁽ⁱ⁾ ~ posterior(· | x_i)` for `i = 1..N_test`.
3. Compute the rank statistic `r_i = P(θ'_d < θ_{i,d} | x_i)`
   per dimension, averaged.
4. Assert that the empirical coverage at each `alpha` is
   within tolerance of the nominal `alpha` (typically 5%).

If the test fails, the SBI posterior is miscalibrated and
must be re-tuned (more simulations, more rounds, different
density estimator) before Phase 2 work begins.

---

## `test_qsvt_*.py`

#### `test_qsvt_approximator.py`

Tests `qsvt/approximator.py`:

- `approximate_threshold(threshold, degree, target_loss, max_loss)`
  returns a list of length `degree`.
- For `degree ≤ 256`, the pyqsp path is used; the result
  agrees with a reference phase list (committed to the
  test file).
- For `degree > 256`, the Chebyshev fallback path is used.
- `QSVTApproximator.approximate_threshold` matches the
  module-level `approximate_threshold` for small degrees.
- `ChebyshevApproximator.compute_phases` returns a list of
  length `degree`.

#### `test_qsvt_circuit.py`

Tests `qsvt/circuit.py`:

- `QSVTRiskCircuit(loader, target_loss, degree, threshold)`
  builds a circuit with `K + 2` qubits.
- The circuit decomposes without error.
- For `degree=2, 4, 8, 16` the circuit has the expected
  number of projector + unitary sandwiches.
- The circuit's `num_qubits` and `K` properties are correct.

#### `test_qsvt_threshold.py`

Tests `qsvt/threshold.py`:

- `ThresholdFunction(threshold, degree, target_loss, max_loss)`
  validates inputs.
- `polynomial_coefficients` returns a `float32` array of
  length `degree + 1`.
- Odd coefficients are zeroed out.
- `evaluate(x)` returns values in `[0, 1]`.
- The threshold function is monotonically non-decreasing in
  `x` (for `x > threshold`).
- `qsvt_phases()` returns a list of length `degree`.

---

## `test_resource_scaling.py`

Tests `experiments/resource_scaling.py`:

- `ResourceEstimate` dataclass has the right fields and
  `as_dict()` returns a plain dict.
- `QuantumResourceEstimator.estimate(K, degree)` returns a
  `ResourceEstimate` with non-negative fields.
- The estimator handles circuit-construction failures by
  falling back to the analytical estimate.
- `compare_loader_vs_gci(K)` returns a dict with both
  estimates.
- `sweep(n_loans_list, degree)` returns a DataFrame with
  one row per K.
- `plot_scaling(df, output_path)` produces a PNG.

---

## CI / smoke runs

The test suite is designed to run in a lightweight CI
environment without GPU, JAX, or `qiskit_aer`. The
following tests are **gated** by `pytest.importorskip` or
`try/except` so they pass in CI:

- `test_data_synthetic.py` — uses NumPy only.
- `test_copula_*.py` — use NumPy + SciPy only.
- `test_compiler_backend_heron.py` — uses Qiskit + mock backends
  only.
- `test_qsvt_*.py` — uses Qiskit + pyqsp; the latter is in
  `requirements.txt` so it should be installed everywhere.
- `test_metrics_*.py` — `test_metrics_quantum_error.py` falls
  back gracefully when `qiskit_aer` is missing.

The following tests require GPU and `qiskit_aer` and may be
skipped in CI:

- `test_phase1_sbc.py` (if a large `N_test` is configured).
- `test_forward_simulator.py` (JAX-GPU path; the NumPy path
  always runs).
- `test_lumi_scaling.py` (Slurm / multi-node tests).

The default `pytest tests/` command runs the lightweight
suite in ~2 minutes on a single CPU core.
