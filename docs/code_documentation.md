# `Code/` — Original QSVT-on-GCI prototype

> Authoritative source of the paper's reference QSVT implementation.
> These files predate the QSVT4CRA research run; we have not rewritten
> them. The research run layers on top via the `loader/`,
> `qsvt/`, and `compiler_backend/` packages.

The `Code/` package contains the four classes that appear in the
paper's headline result:

| File | Symbol | Role |
|---|---|---|
| [`Code/QSVT.py`](#qsvtpy) | `QSVT(QuantumCircuit)` | Core QSVT circuit class |
| [`Code/circuitsCRA.py`](#circuitscrapy) | `get_expected_probability_circuit` | Full risk circuit |
| [`Code/AmplitudeLoading.py`](#amplitudeloadingpy) | `AmplitudeLoading`, `AmplitudeLoadingV2`, `AmplitudeLoadingVar` | Amplitude encoding |
| [`Code/multivariateGCI.py`](#multivariategcipy) | `MultivariateGCI_Poly`, `MultivariateGCI_Linear` | Gaussian Conditional Independence uncertainty model |
| [`Code/utils.py`](#utilspy) | `mapping`, `bisection_search` | Helpers |
| [`Code/generate_dataset.py`](#generatedatasetpy) | CLI | PxWeb → `dataset_regions.py` |
| [`Code/dataset_regions.py`](#datasetregionspy) | data | K=17 Finnish regions, anchor 2024 |
| [`Code/DATASET.md`](#datasetmd) | doc | dataset documentation |

---

## `Code/QSVT.py`

**Class** `QSVT(QuantumCircuit)` — applies the QSVT polynomial transform
to a unitary `U`, sandwiched around an auxiliary qubit. This is the
mathematical heart of the paper.

### Constructor

```python
QSVT(
    unitary_circuit: QuantumCircuit,
    subspace_qubits1: list,
    subspace_qubits2: list,
    phases: list | None = None,
    poly: list | None = None,
    adjust_conventions: bool = False,
    ctrl_zero_qubits1: list = [],
    ctrl_zero_qubits2: list = [],
    name: str = "QSVT",
)
```

Either `phases` or `poly` must be supplied. If `phases is None` and
`poly is not None`, phases are computed via
`pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly, signal_operator="Wx",
method="sym_qsp", measurement="x", chebyshev_basis=True)`.

### `addProj(id_proj: int, phi: float)`

Implements the QSP projector controlled by `subspace_qubits[0]` onto
the auxiliary qubit. The implementation uses a plain `|1><1|`
controlled-RZ sandwich (the previous `X-X` wrap is **gone** — see
the long comment at line 64–78 explaining why this preserves the
final statevector to 1e-9). The actual sequence is:

```
CX subspace → AUX
RZ(2·phi) AUX
CX subspace → AUX
```

With `subspace_qubits` always length 1 in this codebase, a single
`CX` is used (not the 3-CNOT `mcx` decomposition).

### `createQSVT()`

Iterates over `self.rev_phases[:-1]`, calling `addProj` and then
appending `U` (or `U⁻¹` on alternate steps) — the standard
alternating projector / unitary sandwich. The **last** phase is
deliberately *not* applied as a projector (its sign is folded into
the phase-0 modification, see `rev_phases[0] += rev_phases[-1]`).

### `adjust_qsvt_conventions(phases) → list[float]`

Static method. Converts QSP-style phases (in the `W(x)` signal
operator convention) to QSVT conventions used in this paper:

```
phases -= π/2
phases[0]  += π/4
phases[-1] += π/2 + (2·(L-1) - 1) · π/4
return -phases
```

The trailing negative sign flips the convention from
`exp(+iφZ)` to `exp(-iφZ)`.

### `rev_phases` pre-processing

```python
self.rev_phases = phases.copy()
self.rev_phases[0] = self.rev_phases[0] + self.rev_phases[-1]
```

Folds the last phase into the first (the projector's sign is
absorbed into the sandwich; only the projector + unitary sequence
remains in `createQSVT`).

---

## `Code/circuitsCRA.py`

The public entry points used by the notebook and the QPU script.

### `get_expected_loss_circuit(K, uncertainity_model, lgd) → (state_preparation, objective)`

Builds a circuit that prepares the *expected* portfolio loss
`E[loss]` as a measurement of the objective qubit. The scaling
maps `[min(offsets), max(offsets)]` to `[0, 1]` via
`sqrt((x - min)/(max - min))`, so the amplitude on the |1⟩ of
the objective qubit equals the square-root of the *normalised*
expected loss. After post-processing, this recovers
`x · sum(lgd)`.

### `get_cdf_circuit(K, u, x_eval, lgd) → (state_preparation, objective)`

Builds a circuit that prepares the *CDF* at `x_eval`:
`P(loss ≤ x_eval)`. Same plumbing but the objective is the
indicator `1{loss <= x_eval}` per scenario.

### `post_processing(x, lgd) → float`

Converts the quantum amplitude to a loss value: `(x · sum(lgd))`.

### `get_expected_loss_circuitV2(K, uncertainity_model, lgd, c) → (state_preparation, objective)`

Variant with a custom offset `c` and `AmplitudeLoadingV2` (a
multi-controlled-RY decomposition that uses an additional offset
register). Used for higher-accuracy aggregations.

### `get_expected_probability_circuit(K, uncertainity_model, lgd, target_loss, phases=None, poly=None, threshold=0.5, enable_switch=True, epsilon=0.6, verbose=False) → (circuit, objective, ok)`

**The** production risk circuit. Builds the state preparation
(state register + objective register) such that the probability
of measuring `1` on the AUX qubit is `P(loss > target_loss)`
after the QSVT polynomial has been applied.

Internally:

1. Compute `target_loss_scaled = target_loss / sum(lgd)`.
2. Map `[0, target_loss_scaled]` and `[target_loss_scaled, 1]`
   to two halves of the `[0, π/2]` angle range, such that
   `target_loss_scaled ↦ arcsin(threshold)`.
3. Build the `AmplitudeLoadingVar` sub-circuit with the mapped
   offsets.
4. Build the `QSVT` wrapper using `qsvt_application_circuit` with
   `adjust_conventions=True`.

Returns `(state_preparation, objective, True)`. The third return
value is a status flag (always `True` for non-degenerate inputs).

### `qsvt_application_circuit(p_sp, phases=None, poly=None, factor=1) → QSVT`

Thin wrapper around `Code.QSVT.QSVT` that hard-codes the standard
projector-qubit indices (the objective qubit, i.e. the last qubit
of the sub-circuit) and the `ctrl_zero_qubits=[0]` convention
used everywhere in this codebase.

### `build_state_preparation_circuit(uncertainity_model, objective_e_loss, K) → QuantumCircuit`

Glues the uncertainty model and the objective sub-circuit
together, allocating `state` and `objective` registers plus
`objective.num_ancillas` work qubits.

### `build_mcrz(n_qubits, phi, label)`, `build_rz(n_qubits, phi, label)`

Build a multi-controlled RZ and a single-qubit RZ wrapped in a
`UnitaryGate`. These are helper functions used by the older
`qsvt_application_circuit` path; the production path uses
`Code.QSVT.QSVT` directly.

---

## `Code/AmplitudeLoading.py`

Three classes plus two helper gates. They are all
`QuantumCircuit` subclasses that load a real-valued function
`f(j) ↦ α_j` into the rotation angles of a controlled-RY cascade
so that

```
|j⟩|0⟩  ↦  |j⟩(cos(α_j)|0⟩ + sin(α_j)|1⟩)
```

The choice of `cos`/`sin` convention determines whether the
encoded quantity lives on |0⟩ or |1⟩.

### `AmplitudeLoading(num_state_qubits, scaled_values, name="f(x)")`

- `scaled_values` must be in `[-1, 1]`; `len(scaled_values) = 2^K`.
- Computes `alpha_values = 2 · arcsin(scaled_values)`.
- Uses Qiskit's `UCRYGate(alpha_values)`.

This is the original amplitude-loading primitive used by
`get_expected_loss_circuit` (expected-loss / CDF style).

### `AmplitudeLoadingV2(num_state_qubits, scaled_values, c, name="f(x)")`

Variant with a custom offset `c`. Uses
`MultiCRYGateV2(alpha_values, q_reg, starting_offset=π/2 - c)`.

### `AmplitudeLoadingVar(num_state_qubits, scaled_values, starting_offset=0, name="f(x)")`

**This is the one used by the QSVT risk circuit.** Unlike the
other two, it uses `MultiCRGateVar` with an `RY` baseline, and
appends a final `X target_q` at the end so the encoded quantity
ends up on |0⟩ rather than |1⟩. The `starting_offset` is applied
as an `RY(2 · starting_offset)` on the target before the cascade.

- `scaled_values` must be in `[-π, π]`; the helper internally
  doubles them to `[-2π, 2π]` for the `CRY` convention.
- After the cascade: `X` on the target qubit swaps the convention.

### `MultiCRYGateV2(alpha_values, q_reg, starting_offset=0) → QuantumCircuit`

Builds a `CRY` cascade of length `len(q_reg) - 1` plus an
optional starting `RY(starting_offset)` rotation. Each step
applies `CRY(alpha_values[i])` with control `q_reg[i]` and target
`q_reg[-1]`. Returns the *circuit* (not a gate).

### `MultiCRGateVar(alpha_values, q_reg, axis="Y", starting_offset=0) → Gate`

Same idea but returns a `Gate` (so it can be `.to_gate()` and
decomposed). Supports `axis="Y"` (RY/CRY) or `axis="X"` (RX/CRX).

---

## `Code/multivariateGCI.py`

Implements the **Multivariate Gaussian Conditional Independence**
uncertainty model. There are two flavours:

### `_normal_distribution_circuit(n_normal, mu, sigma, bounds) → QuantumCircuit`

A helper that builds a small truncated-normal state-preparation
circuit. Tries to use `qiskit_finance.circuit.library.NormalDistribution`
first; if `qiskit-finance` is not installed (as on many modern
Qiskit 2.x stacks), it falls back to:

1. `grid = linspace(bounds[0], bounds[1], 2^n_normal)`
2. `probs = norm.pdf(grid)`, clipped to non-negative, renormalised.
3. `amplitudes = sqrt(probs) / ||sqrt(probs)||`.
4. State-prep via `StatePreparation` (with a last-resort
   fallback to `qc.initialize`).

For the paper's default `n_normal=2`, the fallback is 2-qubit
state preparation and is very small.

### `MultivariateGCI_Poly(n_normal, normal_max_value, p_zeros, rhos, F_list)`

Polynomial Pauli rotation variant. The class is a `QuantumCircuit`
subclass with `num_qubits = n_normal · sectors + K`:

1. **State prep** — one `_normal_distribution_circuit` per sector,
   creating `sectors` independent latent Z variables on
   `n_normal` qubits each.
2. **Conditional default rotation** — for each loan `k`, a
   `PolynomialPauliRotations(n_normal, [offset, slope, 0, cubic, 0, quintic])`
   encoding the per-loan default probability as a Taylor expansion
   of the probit link
   `P(default_k | Z) = Φ((Φ⁻¹(p_k) - ρ·b_k·Z)/sqrt(1-ρ))`.
3. The result lives on `n_normal · sectors + K` qubits; the
   last K qubits hold per-loan default indicators.

The polynomial coefficients are derived from a third-order Taylor
expansion of `Φ(Φ⁻¹(p)/sqrt(1-ρ))` around `z=0`, scaled by the
sector loadings `F[i]`.

### `MultivariateGCI_Linear(n_normal, normal_max_value, p_zeros, rhos, F_list)`

Same architecture, but with `LinearPauliRotations(n_normal, slope, offset)`
instead of `PolynomialPauliRotations`. **This is the variant used
in production** (`experiments/boston_qpu.py`); it transpiles to
fewer 2-qubit gates on Heron and is the recommended
uncertainty-model for `K ≤ 10`.

**Known scaling limitation**: the model uses `2^n_normal` qubits
per sector. For `n_normal=2` (the default in this paper), each
sector is 4 qubits; for `K=17` plus `n_z=2` sectors, the state
qubits alone are `2·2 + 17 = 21`, which is small enough for
near-term hardware. The research run's `loader/posterior_factor_copula.py`
removes this scaling bottleneck by replacing the
`NormalDistribution`-based encoding with amplitude loading
(see [`docs/loader_documentation.md`](loader_documentation.md)).

---

## `Code/utils.py`

Two helpers.

### `mapping(decimal_number, lgd, K) → float`

Treats `decimal_number` as a K-bit binary default pattern (LSB
first) and returns `sum(lgd[i] for bit i where the i-th bit is 1)`.
This is the total portfolio loss for that scenario.

### `bisection_search(objective, target_value, low_level, high_level, low_value=0, high_value=1, sampler=None, phis=None, rescaling_factor=1) → dict`

Classical bisection search over a quantum objective. Used to
solve for the value `ℓ` such that the quantum circuit evaluates
`P(loss ≤ ℓ) = target_value`. Stops when `high_level - low_level ≤
total_loss/1000`. Returns `{level, value, num_eval, comment}`.

This is the "classical outer loop" that drives the quantum
inner loop in the paper's headline protocol. In the QSVT4CRA
research run, the bisection is replaced by a direct
QSVT-evaluation at a single threshold loss; see
[`docs/qsvt_documentation.md`](qsvt_documentation.md).

---

## `Code/generate_dataset.py`

CLI: `python Code/generate_dataset.py` (no arguments). Fetches
the 2024 Statistics Finland PxWeb tables `157w` (indebtedness)
and `13al` (unemployment) for 17 mainland regions, then writes
[`Code/dataset_regions.py`](#datasetregionspy).

Data sources:

- **Table 157w** — `StatFin/velk/statfin_velk_pxt_157w.px` — per-region
  number of housing-indebted households, mean debt, mean
  interest, share indebted.
- **Table 13al** — `StatFin/tyti/statfin_tyti_pxt_13al.px` —
  per-region unemployment rate, 2024.

External assumptions (documented in the module docstring):

| Parameter | Value | Source |
|---|---|---|
| `PD_BASE` | 0.015 | EBA Risk Dashboard (FI residential NPL band 0.5–1.5%) |
| `LGD_RATE` | 0.20 | EBA EU-wide stress test baseline |
| `RHO` | 0.09 | QSVT4CRA paper default |
| `F_SCALE` | 0.30 | Constraint `sqrt(sum(F^2)) < 1` from GCI model |

The output `dataset_regions.py` has the exact format the
`CRA_QSVT.ipynb` notebook cell 2 expects.

### Functions

- `pxweb_query(path, body) → list[dict]` — POST to the
  PxWeb REST API, return `data` rows.
- `fetch_157w() → dict[region → {n_indebted, share_pct, mean_debt, mean_int}]`.
- `fetch_unemployment() → dict[region → rate]`.
- `zscore(xs) → list[float]` — population-mean and population-stddev
  z-score (uses `statistics.pstdev`, divides by N, not N-1).
- `compute_parameters(debt_data, unemp) → dict` — assembles the
  per-region parameters, fills missing unemployment with the
  national mean.
- `emit_module(out, path)` — writes a Python module on disk with
  the exact format the notebook expects.
- `main()` — CLI entry point.

### Notes

- The PxWeb endpoint paths reflect the **8 June 2026** database
  identifier change (new short codes like `StatFin/velk/...`
  instead of `StatFin__velk/...`).
- Kainuu (MK18) is reported as `"."` in 2024 and is filled with
  the 17-region mean.
- The script requires internet access to pxdata.stat.fi.

---

## `Code/dataset_regions.py`

**Auto-generated by `Code/generate_dataset.py`.** Do not edit
by hand; re-run the generator to refresh.

Constants (committed snapshot from 2026-06-08):

```python
K = 17
n_z = 2
z_max = 2

regions = ["Uusimaa", "Southwest Finland", ..., "Lapland"]  # 17 names
region_codes = ["MK01", "MK02", ..., "MK19"]              # 17 codes

p_zeros = [0.01582, 0.01516, ..., 0.01394]  # length 17
rhos    = [0.09] * 17                       # constant
lgd     = [8.235e9, 1.578e9, ..., 4.081e8]  # EUR, length 17
F_values = [[0.041, 0.2972], ...]            # 17 × 2 latent loadings

unemployment_rate_2024_pct = {"MK01": 8.9, ...}
mean_debt_eur = {"MK01": {"name": "Uusimaa", "n_indebted": 260068, ...}, ...}
```

Sum of `lgd` is approximately **€18.4 bn** (upper bound of
portfolio loss in the worst-case all-regions-default scenario).

---

## `Code/DATASET.md`

Documentation for the K=17 dataset. Lists:

- Region selection rationale (17 mainland, drops MK16 small,
  MK21 Åland).
- Per-parameter provenance table (PD baseline, LGD rate, etc.).
- Five caveats:
  1. PDs are approximated, not measured.
  2. LGDs use a single 20% rate.
  3. F_values are standardised, not estimated correlations.
  4. K=17 → 2^17=131 072 scenarios (runtime warning).
  5. Kainuu unemployment filled with national mean.

Plus a reproducibility section (`python Code/generate_dataset.py`).
