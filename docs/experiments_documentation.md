# `experiments/` — CLI entry points

> CLI scripts for each research-run phase. All entry points are
> runnable as `python -m experiments.<name>`. They are the
> "outer loop" of the QSVT4CRA pipeline.

## Files

| File | Phase | Purpose |
|---|---|---|
| [`experiments/sbi_train.py`](#sbi_trainpy) | Phase 1 | SBI training CLI |
| [`experiments/mc_ground_truth.py`](#mc_ground_truthpy) | Phase 3 | Massive-MC ground truth |
| [`experiments/qsvt_sweep.py`](#qsvt_sweeppy) | Phase 4 | QSVT degree sweep |
| [`experiments/boston_qpu.py`](#boston_qpupy) | Phase 4 | Real IBM QPU benchmark |
| [`experiments/heron_simulation.py`](#heron_simulationpy) | Phase 4 | Aer-on-Heron calibration simulation |
| [`experiments/analyze_degree_sweep.py`](#analyze_degree_sweeppy) | Phase 4 | Post-process degree sweep results |
| [`experiments/check_qpu_jobs.py`](#check_qpu_jobspy) | Phase 4 | Job-status checker |
| [`experiments/ood_robustness.py`](#ood_robustnesspy) | Phase 5 | Distribution shift experiment |
| [`experiments/resource_scaling.py`](#resource_scalingpy) | Phase 6 | Quantum resource estimation |
| [`experiments/make_figures.py`](#make_figurespy) | Phase 7 | 8 publication figures + poster |
| [`experiments/__init__.py`](#initpy) | — | Marker file (one-line docstring) |

---

## `sbi_train.py`

Phase 1: train an SBI posterior over factor-copula parameters.
Three estimators (`npe`, `nle`, `fm`).

### CLI

```bash
python -m experiments.sbi_train \
    --method npe \
    --n-simulations 1000 \
    --n-rounds 10 \
    --device cuda \
    --K 10 \
    --regime baseline \
    --output checkpoints/sbi_npe_baseline.pt
```

### Key functions

#### `_build_prior(K, regime) → (low, high)`

Constructs prior lower/upper bounds for the
factor-copula parameter vector. Index layout is the standard
`2K+4`:

- `theta[:K]` factor loadings in `[-0.7, 0.7]`.
- `theta[K:2K]` default probabilities in `[0.001, 0.30]` for
  baseline, `[0.001, 0.50]` for stress regimes.
- `theta[2K]` tail dependence in `[0, 1]`.
- `theta[2K+1:2K+4]` copula params `(rho, nu, rotation)` in
  `[(0, 4, -1), (0.95, 100, 1)]`.

#### `_make_simulator(K, regime, seed) → NumPyForwardSimulator`

Builds a forward simulator with the right K and seed.

#### `_simulate_training_pairs(simulator, low, high, n, seed) → list[(theta, x)]`

Samples `n` `theta` vectors from the uniform hyperrectangle
`[low, high]`, runs the simulator on each in batch mode
(`simulate(theta_batch, n_scenarios=1)`), and returns the
list of `(theta, x)` pairs.

The `n_scenarios=1` is intentional: the SBI observation
vector is a 10-column summary, not a per-scenario loss
array. This means each training pair is a single
summary-statistic observation, which is the standard
formulation for amortised SBI.

#### `_train_npe / _train_nle / _train_fm(simulator, low, high, n_simulations, n_rounds, device, seed, wandb_project) → TrainingResult`

Dispatch wrappers that build the right pipeline and call
`.train()`. All three follow the same signature.

#### `_save_checkpoint(result, method, K, regime, n_simulations, n_rounds, seed, output)`

Saves the training result to a torch checkpoint. The
checkpoint dict contains:

- `method`, `K`, `regime`, `n_simulations`, `n_rounds`, `seed`.
- `posterior_samples`: a `np.ndarray` of 1000 samples drawn
  from the trained posterior.
- `log_probs`, `ess_values`: training diagnostics.
- `wandb_url`: optional W&B URL.

If sampling from the posterior fails, `posterior_samples` is
an empty array of shape `(0, 2*K+4)`.

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--method` | `npe` | `npe`, `nle`, or `fm` |
| `--K` | `10` | Number of loans |
| `--regime` | `baseline` | One of `baseline`, `housing_crash`, `rate_shock_0.5`, `rate_shock_1.5`, `unemployment`, `liquidity` |
| `--n-simulations` | `1000` | Training pairs to generate |
| `--n-rounds` | `10` | Proposal-adaptation rounds |
| `--device` | `cuda` if available else `cpu` | `cuda` or `cpu` |
| `--seed` | `42` | |
| `--wandb-project` | `None` | W&B project name |
| `--output` | `checkpoints/sbi_npe.pt` | Checkpoint path |
| `--time-limit` | `None` | Warns if training exceeds this; not enforced |

The script also forces CPU if `--device=cuda` is requested
but no CUDA is available, with a clear warning.

---

## `mc_ground_truth.py`

Phase 3: massive-MC ground truth for VaR/CVaR benchmarking.

### CLI

```bash
python -m experiments.mc_ground_truth \
    --posterior-checkpoint checkpoints/sbi_npe_baseline_K10.pt \
    --n-scenarios 1000000 \
    --copula gaussian \
    --regime baseline \
    --K 10 \
    --output results/ground_truth_gaussian_baseline.npz
```

### Key functions

#### `_load_posterior(path) → np.ndarray`

Loads posterior samples from `.npy`, `.npz`, or `.pt` files.
For `.npz`, looks for keys `posterior`, `samples`, or `theta`.
For `.pt` (PyTorch), looks for the same keys. Reshapes 1D
arrays to `(1, D)`.

#### `_get_copula(name, K, seed) → FactorCopula`

Returns a `GaussianFactorCopula` or `StudentTFactorCopula`
by name. Raises `ValueError` for unknown names.

#### `main(argv=None)`

The CLI. Validates that either `--posterior-checkpoint` or
`--posterior-samples` is provided, builds the copula and
portfolio generator, runs `GroundTruthMC.run()` (or
`run_streaming()` if `--streaming`), prints a summary table,
and saves results to a `.npz` file.

### Output file

The output `.npz` contains:

- `posterior_var`, `posterior_cvar`, `posterior_var_99`,
  `posterior_cvar_99`, `posterior_var_999`,
  `posterior_cvar_999`: per-sample metrics.
- `predictive_var_at_*`, `predictive_cvar_at_*`:
  posterior-mean metrics.
- `n_posterior_samples`, `n_scenarios_per_posterior`, `regime`,
  `runtime_seconds`, `copula`: metadata.

The `all_loss_samples` array from `GroundTruthMC.run()` is
**not** saved by default (memory guard).

---

## `qsvt_sweep.py`

Phase 4: QSVT polynomial degree sweep.

### CLI

```bash
python -m experiments.qsvt_sweep \
    --degrees 16 32 64 128 256 512 1024 \
    --posterior-checkpoint posterior_samples.npz \
    --output results/qsvt_sweep.npz \
    --K 10 \
    --target-loss 0.5 \
    --n-shots 10000
```

### Behaviour

- If `--posterior-checkpoint` is given, loads samples from
  `.npy` / `.npz` / `.pt`. Keys: `theta_samples`,
  `posterior_samples`, or any 2D array.
- Uses the **first** posterior sample for the sweep.
- Computes a classical `var_0_95` baseline from
  `GaussianFactorCopula.sample(theta, n_samples=100_000)`.
- For each degree, builds
  `PosteriorFactorCopulaLoader → QSVTRiskCircuit`, runs
  `quantum_vs_classical_error` against the normalised
  classical value, and records the result.
- Saves everything to a `.npz` file with keys like
  `d16_quantum_estimate`, `d16_abs_error`, `d16_circuit_depth`,
  etc.

### Output file

A `.npz` with:

- `degrees`, `K`, `target_loss`, `n_shots`,
  `classical_var_95`, `classical_cvar_95`, `runtime_seconds`.
- For each degree `d`: `d{deg}_quantum_estimate`,
  `d{deg}_classical_value`, `d{deg}_abs_error`,
  `d{deg}_rel_error`, `d{deg}_ci_95_lower`,
  `d{deg}_ci_95_upper`, `d{deg}_circuit_depth`,
  `d{deg}_num_qubits`. If a degree failed, the corresponding
  `d{deg}_error` is a string with the error message.

---

## `boston_qpu.py`

Phase 4: compile, submit, and benchmark the K=17 Finnish
mortgage QSVT circuit on the real IBM `ibm_boston` (Heron r3)
QPU. The most operationally complex experiment in the
project.

### What it does

1. Builds the QSVT circuit with **real Chebyshev threshold
   phases** from `qsvt.approximator.approximate_threshold`
   (via `pyqsp`). The all-zero "smoke" phases from
   `heron_simulation.py` are explicitly avoided.
2. Computes a classical GCI Monte-Carlo reference for the
   same loss distribution.
3. Fetches `ibm_boston` and compiles the circuit with the
   calibration-aware Heron pass manager.
4. Submits to the real QPU via
   `qiskit_ibm_runtime.SamplerV2`. Optionally submits two
   calibration circuits in the same batch for per-qubit
   readout mitigation.
5. Recovers the **AUX** marginal counts (the QSVT projector
   answer bit, **not** the Target qubit).
6. Optionally applies per-qubit readout mitigation via
   `C^{-1} @ p_obs`.
7. Writes a full payload to the output JSON.

### Key functions

#### `_load_finnish_mortgage_dataset(k) → dict`

Loads `Code.dataset_regions`, optionally truncated to the
first `k` regions. Returns a dict with keys
`K`, `regions`, `lgd`, `p_zeros`, `rhos`, `F_values`,
`n_z`, `z_max`, `source`.

#### `_synthetic_gaussian_dataset(K, n_z=2, z_max=2, seed=0, p_zero_range=(0.005, 0.02), rho=0.09, lgd_range=(1e8, 1e9)) → dict`

Builds a synthetic K-region Gaussian factor-copula dataset
for benchmarks that aren't tied to the Finnish data.

#### `build_qsvt_circuit_from_dataset(dataset, degree, target_loss_fraction) → (circuit, meta)`

Builds the QSVT circuit for an arbitrary dataset dict:

1. Constructs a `MultivariateGCI_Linear` uncertainty model
   from `dataset['p_zeros']`, `dataset['rhos']`,
   `dataset['F_values']`.
2. Computes `max_loss = sum(lgd)` and `target_loss =
   target_loss_fraction * max_loss`.
3. Computes phases via `approximate_threshold(threshold=0.5,
   degree=degree, target_loss=0.5, max_loss=1.0)`.
4. Calls
   `Code.circuitsCRA.get_expected_probability_circuit(...)` to
   assemble the full circuit.
5. Builds a `meta` dict with the AUX / target qubit indices
   and all dataset scalars.

#### `build_finnish_mortgage_qsvt_circuit(degree, target_loss_fraction, k=17) → (circuit, meta)`

Thin wrapper around `build_qsvt_circuit_from_dataset` for
the Finnish mortgage data.

#### `classical_gci_losses(p_zeros, rhos, f_values, lgd, n_scenarios, n_z, z_max, seed) → np.ndarray`

Samples portfolio losses from the GCI model encoded by
`Code.multivariateGCI.MultivariateGCI_Linear`. The latent
`Z ~ N(0, I_{n_z})` is optionally truncated to `[-z_max,
+z_max]`. Per-region default probability is
`PD_i(z) = Φ((Φ⁻¹(p_i) - ρ·F_i·z) / sqrt(1-ρ))`.

#### `classical_reference(meta, n_scenarios, alphas, seed) → dict`

Computes classical VaR/CVaR/tail-probability reference via
`metrics.var_cvar.var_cvar` plus a tail-probability
calculation.

#### `_extract_bit_counts(bit_array_or_data, bit_index) → dict`

Parses SamplerV2 primitive results. Accepts both
`qiskit.primitives.containers.BitArray` and `DataBin`
(SamplerV2 result in qiskit 2.x). For multi-bit registers,
slices to `bit_index`.

#### `_mitigate_single_qubit(counts, confusion) → dict`

Applies per-qubit readout mitigation: `p_true = C^{-1} @
p_obs`. Returns the corrected counts (rounded to int) plus
`_mitigated` and `_condition_number` flags.

The confusion matrix convention is
`C[i, j] = P(measure=i | state=j)`, so columns are the two
"prep then measure" calibration columns.

#### `_build_calibration_circuits(aux_physical_qubit, cal_shots, backend) → (cal_0, cal_1)`

Pre-transpiles the `|0⟩` and `|1⟩` prep calibration circuits
onto the AUX's physical qubit.

#### `submit_main_and_mitigate(compiled_circuit, backend, shots, aux_qubit_index, apply_mitigation, cal_shots=1024, report=None) → (counts_mit, counts_raw, calib_payload, job_id)`

Submits the main experiment + (optional) calibration circuits
as separate *pubs* of a single `SamplerV2.run()` call.
Batching them ensures they run back-to-back and avoids
"stuck in queue" issues from sequential submissions.

#### `_resolve_aux_physical_qubit(compiled_circuit, aux_qubit_index, report=None) → int`

Returns the IBM physical qubit ID that hosts the AUX. Tries
the transpiled circuit's own `layout` first
(`layout.get_virtual_bits()` for Qubit-object-aware lookup),
then falls back to `report.selected_physical_qubits[aux_qubit_index]`.

#### `run_boston_qpu(backend_name, degree, shots, target_loss_fraction, n_classical_scenarios, classical_seed, output, api_key_env, channel, optimization_level, seed_transpiler, apply_mitigation, cal_shots, token_file, use_pyzx, k, dataset_source, synthetic_seed, synthetic_n_z) → dict`

The top-level pipeline. Returns the full payload as a dict
(the same dict that is written to the output JSON). See the
`main` CLI section below for the full list of arguments.

### CLI

```bash
python -m experiments.boston_qpu \
    --backend-name ibm_boston \
    --dataset-source finnish \
    --k 17 \
    --degree 8 \
    --shots 2048 \
    --target-loss-fraction 0.5 \
    --n-classical-scenarios 200000 \
    --classical-seed 0 \
    --output results/boston_qpu_k17.json \
    --no-mitigation
```

Notable flags:

- `--dataset-source {finnish,gaussian}` — choose real or
  synthetic data.
- `--k` — number of regions (1–17 for finnish, ≥1 for
  gaussian).
- `--degree` — QSVT polynomial degree (default 8; the
  sweet spot before noise dominates on Heron).
- `--use-pyzx` — opt into the `pyzx.basic_optimization`
  round-trip.
- `--no-mitigation` — skip the per-qubit readout calibration
  and mitigation round.
- `--cal-shots` — shots per calibration prep circuit
  (default 512).
- `--token-file` — path to a local token file; pass `""` to
  disable the file fallback and use only `$IBM_API_KEY`.

### Output file

A JSON with the full payload:

```json
{
  "backend_name": "ibm_boston",
  "degree": 8,
  "shots": 2048,
  "apply_mitigation": true,
  "job_id": "...",
  "dataset": { ... K, regions, lgd, p_zeros, rhos, F_values, n_z, z_max, target_loss, target_loss_fraction, target_qubit_index, aux_qubit_index, phases, data_source ... },
  "precompile": { "num_qubits": ..., "depth": ..., "count_ops": {...} },
  "compile_report": { "backend_name", "processor_type", "n_qubits", "depth", "two_qubit_depth_proxy", "gate_counts", "selected_physical_qubits", "estimated_success_probability", "notes" },
  "qpu_counts_raw": {"0": ..., "1": ...},
  "qpu_counts_mitigated": {"0": ..., "1": ...},
  "readout_calibration": { "applied": true, "cal_shots", "cal_physical_qubit", "c0_counts", "c1_counts", "e10_meas1_given_state0", "e01_meas0_given_state1", "confusion_matrix" },
  "qpu_total_shots": ...,
  "qpu_p_loss_gt_target_raw": ...,
  "qpu_p_loss_gt_target": ...,
  "qpu_ci_95_p_loss_gt": [lower, upper],
  "classical": { "var_0_95", "var_0_99", "var_0_999", "cvar_0_95", ..., "tail_at_target_loss", "n_scenarios", "max_loss" },
  "benchmark": {
    "metric": "P(loss > target_loss)",
    "classical_value": ...,
    "quantum_value_raw": ...,
    "quantum_value": ...,
    "abs_error_raw": ...,
    "rel_error_raw": ...,
    "abs_error": ...,
    "rel_error": ...
  },
  "runtime_seconds": ...
}
```

The CLI summary printed at the end of a run is a small
projection: backend, job_id, compiled depth, raw and
mitigated P(loss > target), classical reference, errors, and
readout e10.

---

## `heron_simulation.py`

Phase 4: compile and Aer-simulate the K=17 Finnish mortgage
QSVT circuit using Heron calibration data. **Does not
submit to a real QPU.**

The main difference from `boston_qpu.py` is that this script
uses **all-zero "smoke" phases** (`phases = [0.0] *
max(2, int(degree))`), which makes the QSVT projector the
identity. This is intentional — the goal of
`heron_simulation.py` is to verify compilation, layout, and
noisy-run feasibility, not to benchmark the QSVT
mathematics.

`build_finnish_mortgage_qsvt_circuit(degree=4, target_loss_fraction=0.5)` builds the smoke circuit.

The CLI is identical to `boston_qpu.py` except for the
flags `--measure-all` and `--noiseless` (which use the
ideal `AerSimulator` instead of `AerSimulator.from_backend`).

Output JSON has the same structure as `boston_qpu.py` minus
the calibration / mitigation fields.

---

## `analyze_degree_sweep.py`

Post-process the results of a degree sweep that was
submitted earlier.

### What it does

1. Reads `results/degree_sweep_ids.json` (a `{degree: {job_id,
   aux_phys}}` mapping).
2. For each degree, fetches the job result from IBM Runtime.
3. Extracts the main + two calibration pub counts.
4. Computes the per-qubit confusion matrix and applies
   `_mitigate_single_qubit` to the main counts.
5. Runs an Aer simulation of the **same circuit** (using
   the IBM backend as the noise model source via
   `AerSimulator`) as a noiseless reference.
6. Computes the classical `P(loss > target_loss)` via
   `experiments.boston_qpu.classical_reference`.
7. Writes a `results/degree_sweep_summary.json` with all
   the per-degree rows, and prints a trade-off table:

```
  deg  Aer_id     QPU_raw    QPU_mit    class      rel_err_mit
  ----  ----       -------    -------    -------    -----------
  4     0.4982     0.5124     0.4976     0.4912     0.013
  8     0.5021     0.5189     0.4998     0.4912     0.018
  ...
```

### CLI

```bash
python -m experiments.analyze_degree_sweep \
    --ids-file results/degree_sweep_ids.json \
    --output results/degree_sweep_summary.json \
    --n-classical-scenarios 200000
```

---

## `check_qpu_jobs.py`

Status checker for recently submitted IBM jobs.

### CLI

```bash
# List recent jobs (default 10)
python -m experiments.check_qpu_jobs

# Filter by backend, more entries
python -m experiments.check_qpu_jobs --backend ibm_boston --limit 20

# Fetch and persist the result of a specific completed job
python -m experiments.check_qpu_jobs --fetch <job_id> \
    --output results/qpu_job_summary.json
```

### What it does

- **List mode**: queries `service.jobs(limit=10, descending=True)`
  and prints a table of `{JOB_ID, STATUS, BACKEND, NAME}`.
  Filters by `--backend` if given.
- **Fetch mode**: looks up a specific job by ID, waits for
  status `DONE` or `COMPLETED`, extracts the per-pub marginal
  counts (`{"0": n0, "1": n1}` and the P(bit=1) value), and
  writes a summary JSON.

This script is the right tool for polling long-queue QPU
jobs after submission.

---

## `ood_robustness.py`

Phase 5: out-of-distribution robustness experiment.

### What it does

For each test regime (e.g., `baseline`, `housing_crash`,
`rate_shock_0.5`, `rate_shock_1.5`, `unemployment`,
`liquidity`):

- **Method A — point-estimate GCI**: θ̂ = posterior mean,
  apply regime shock, compute VaR/CVaR via copula.
- **Method B — posterior-propagated factor copula**: for each
  θ⁽ⁱ⁾ in `posterior_samples`, apply regime shock, compute
  per-sample VaR/CVaR, aggregate (mean ± std).

The output is a `.npz` with:

- `regime_results`: per-regime dicts with both methods'
  metrics (per-sample VaR/CVaR arrays + mean/std summaries).
- `summary`: aggregate metrics
  (`method_A_avg_tail_coverage_error`,
  `method_B_avg_tail_coverage_error`,
  `method_B_uncertainty_widening`).

If `--plot` is given, a 2x2 matplotlib figure is also saved
to that path:

1. **Panel 1**: VaR95 by regime (Method A bar, Method B bar
   with error bars).
2. **Panel 2**: Tail coverage by regime (Method A line,
   Method B dashed line, 5% reference line).
3. **Panel 3**: Posterior uncertainty widening (log-scale
   bar chart, baseline width = 1.0).
4. **Panel 4**: VaR calibration error (mean |empirical α -
   nominal α|) by regime.

### Key classes

#### `OODExperiment(posterior_samples, train_regime, test_regimes, n_scenarios, K, copula, seed)`

Stateful experiment runner. The `run()` method does the
per-regime two-method comparison and stores results in
`self._results`. `save_npz(output_path)` serialises to disk.

#### `compare_methods_plot(results, output_path)`

Generates the 2x2 comparison figure.

### CLI

```bash
python -m experiments.ood_robustness \
    --posterior-checkpoint checkpoints/sbi_npe_baseline_K10.pt \
    --test-regimes baseline housing_crash rate_shock_0.5 rate_shock_1.5 unemployment liquidity \
    --n-posterior-samples 1000 \
    --n-scenarios 100000 \
    --K 10 \
    --output results/ood_robustness_K10.npz \
    --plot figures/ood_robustness_K10.png
```

### Regime naming

Special cases (in `_REGIME_SHOCK_MAP`):

- `rate_shock_0.5` → `rate_shock` with `shock=0.5` (low).
- `rate_shock_1.5` → `rate_shock` with `shock=1.5` (high).

Any other name is treated as a base regime name with
`shock=1.0`.

---

## `resource_scaling.py`

Phase 6: quantum resource estimation. Estimates qubits,
depth, T-count, and 2-qubit depth as a function of portfolio
size K.

### Data classes

#### `ResourceEstimate(K, n_qubits, n_state_qubits, n_ancilla_qubits, qsvt_degree, circuit_depth, t_count, cz_count, rz_count, mcz_count, two_qubit_depth, gci_compatible, memory_mb_estimate, estimated_aer_runtime_s, infeasible=False, notes="")`

The per-K resource summary. `as_dict()` returns a plain dict
for DataFrame / JSON serialisation.

#### `QuantumResourceEstimator(loader_factory, qsvt_factory=None, target_loss=0.5, default_degree=64)`

The estimator. Subclasses / callers provide two factories:

- `loader_factory(K) → PosteriorFactorCopulaLoader`.
- `qsvt_factory(loader, target_loss, degree) → QuantumCircuit`
  (optional; falls back to `QSVTRiskCircuit` if available).

Methods:

- `estimate(K, target_loss=None, degree=64) → ResourceEstimate`.
  Builds the loader, builds the full circuit (via the QSVT
  factory or fallback), transpiles to basis gates
  (`cx, rz, x, sx`), extracts gate counts, computes T-count
  estimate `4 * (rz + cz + mcz)`, two-qubit depth proxy,
  state-vector memory
  (`16 bytes * 2^n_qubits`),
  Aer runtime estimate, GCI compatibility (`K <= 10`), and
  infeasibility (`n_qubits > 30`).
- `_estimate_analytical(K, degree, target_loss, error_msg) → ResourceEstimate`.
  Analytical fallback when circuit construction fails
  (e.g., K too large for the histogram approach in
  `PosteriorFactorCopulaLoader`):
  `n_qubits = K + 2`, `circuit_depth = K * degree * 3`,
  `t_count = 4 * (rz + cz + mcz)`.
- `compare_loader_vs_gci(K) → dict`.
  Returns `{"K": K, "copula": <ResourceEstimate>,
  "gci": <ResourceEstimate>}`.
- `sweep(n_loans_list, degree=64) → pd.DataFrame`.
  Runs `estimate` for each K and returns a DataFrame.
- `plot_scaling(df, output_path) → None`.
  Generates a 4-panel plot (n_qubits / circuit_depth /
  T-count / Aer runtime, all vs K).

### CLI

```bash
python -m experiments.resource_scaling \
    --n-loans 10 50 100 500 1000 \
    --degree 64 \
    --output figures/resource_scaling.png \
    --csv results/resource_scaling.csv
```

---

## `make_figures.py`

Phase 7: generates 7 publication figures + 1 hackathon
poster figure.

### Figure plan

| # | Title | Method |
|---|---|---|
| 1 | Posterior uncertainty over copula parameters | `figure1_posterior_uncertainty` |
| 2 | Loss distributions — GCI vs posterior factor copula | `figure2_loss_distributions` |
| 3 | VaR/CVaR uncertainty bands | `figure3_var_cvar_uncertainty` |
| 4 | QSVT approximation error vs degree | `figure4_qsvt_error` |
| 5 | OOD calibration comparison | `figure5_ood_calibration` |
| 6 | Quantum resource scaling | `figure6_quantum_scaling` |
| 7 | End-to-end pipeline (HACKATHON POSTER) | `figure7_pipeline_poster` |

### `FigureGenerator(results_dir, output_dir, K=10, seed=42)`

Generates all 8 figures. Constructor builds a "synthetic
fallback" dataset (used when real results are absent) so the
method works even before Phase 1–6 have run.

`run_all(synthetic_fallback=True) → dict[str, Path]` generates
all 8 figures and returns a mapping from figure name to
saved PNG path. Each `figureN_*` method is called
independently; failure of one figure does not abort the
rest.

`save_figure(fig, name, dpi=300) → Path` is a helper that
saves a matplotlib figure to `<output_dir>/<name>.png` at the
requested DPI.

### CLI

```bash
python -m experiments.make_figures \
    --results-dir results/ \
    --output-dir figures/ \
    --K 10 \
    --dpi 300
```

### Color palette

The module uses a fixed colour palette (see `PALETTE`
constant) to keep all 8 figures visually consistent.

---

## `__init__.py`

Marker file. Contains a single one-line docstring:

```python
"""experiments — CLI entry points for QSVT4CRA research runs."""
```

This is needed for `python -m experiments.<name>` to work
(the directory must be a Python package).
