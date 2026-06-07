# `qsvt/` — QSVT circuits and approximators

> The QSVT layer. Wraps `Code/QSVT.py` and `pyqsp` with a
> high-level interface tailored to the risk-circuit use case.

## Files

| File | Public API | Purpose |
|---|---|---|
| [`qsvt/circuit.py`](#circuitpy) | `QSVTRiskCircuit` | Full QSVT risk circuit composition |
| [`qsvt/approximator.py`](#approximatorpy) | `QSVTApproximator`, `ChebyshevApproximator`, `approximate_threshold` | QSVT phase-sequence computation |
| [`qsvt/threshold.py`](#thresholdpy) | `ThresholdFunction` | Even threshold function polynomial |

The package exposes `QSVTRiskCircuit` as the only top-level
import:

```python
from qsvt import QSVTRiskCircuit
```

---

## `circuit.py`

**Class** `QSVTRiskCircuit(QuantumCircuit)`.

Composes `PosteriorFactorCopulaLoader` + threshold amplitude
loading + `QSVT` (from `Code/QSVT.py`) into a complete circuit
for VaR/CVaR computation. Architecture decision D4.

### Constructor

```python
QSVTRiskCircuit(
    loader: PosteriorFactorCopulaLoader,
    target_loss: float,
    degree: int,
    threshold: float = 0.5,
    name: str = "QSVTrisk",
)
```

- `loader`: a `PosteriorFactorCopulaLoader` instance providing
  the state preparation.
- `target_loss`: VaR/CVaR threshold (in loss units).
- `degree`: QSVT polynomial degree.
- `threshold`: value in `[0, 1]` for the step function
  (default 0.5; the loss > target maps to > threshold
  amplitude).

### Registers

The class allocates 3 quantum registers:

| Register | Qubits | Purpose |
|---|---|---|
| `state` | K | Loss-distribution state from loader |
| `target` | 1 | Encoded loss value (via `AmplitudeLoadingVar`) |
| `aux` | 1 | QSVT projector answer bit (P(loss > target_loss)) |

Total `num_qubits = K + 2`.

### `_build_circuit(K, lgd, target_loss, degree, threshold)`

1. Appends `loader.to_gate()` on `qubits[:K+1]`.
2. Computes phases via `approximate_threshold`.
3. Builds the objective sub-circuit via
   `_build_objective_circuit` (an `AmplitudeLoadingVar` that
   maps loss values to angles).
4. Wraps the objective with `Code.QSVT.QSVT(...)` using
   `subspace_qubits=[K]` (the target qubit is the projector
   control).
5. Appends the QSVT gate to the full circuit.

### `_build_objective_circuit(K, lgd, target_loss, threshold) → QuantumCircuit`

Builds the `AmplitudeLoadingVar` sub-circuit with the same
loss-to-angle mapping as
`Code/circuitsCRA.get_expected_probability_circuit`:

1. `target_loss_scaled = target_loss / sum(lgd)`.
2. Map `[0, target_loss_scaled]` and `[target_loss_scaled, 1]`
   to two halves of `[0, π/2]` so that
   `target_loss_scaled ↦ arcsin(threshold)`.
3. Compute `scaled_offsets[j] = transform(loss_j) - transform(0)`.
4. Construct `AmplitudeLoadingVar(K, scaled_offsets,
   starting_offset=transform(0))`.

### Properties

- `num_qubits` — total qubits in the circuit.
- `K` — number of state qubits (delegated to the loader).

### Usage example

```python
import numpy as np
from loader.posterior_factor_copula import PosteriorFactorCopulaLoader
from qsvt.circuit import QSVTRiskCircuit

K = 10
theta = np.zeros(2 * K + 4, dtype=np.float32)
theta[:K] = 0.3
theta[K:2*K] = 0.02

loader = PosteriorFactorCopulaLoader(theta=theta, K=K, max_loss=K * 0.4)
circuit = QSVTRiskCircuit(loader, target_loss=2.0, degree=16)
print(circuit.num_qubits)  # K + 2 = 12
```

---

## `approximator.py`

Top-level helper for QSVT phase-sequence computation. Three
classes plus one convenience function.

### `QSVTApproximator(prior: dict | None = None)`

Thin wrapper around `ThresholdFunction` /
`pyqsp.angle_sequence.QuantumSignalProcessingPhases`.

- `approximate_threshold(threshold, degree, target_loss=0.5, max_loss=1.0) → list[float]`
  Builds a `ThresholdFunction` and returns `tf.qsvt_phases()`.

### `ChebyshevApproximator(degree: int)`

Hand-rolled Chebyshev phase-sequence generator. Used as a
fallback when `pyqsp` fails at `degree ≥ 256`.

- `compute_phases(poly_coeffs, threshold=0.5) → list[float]`
  1. L2-normalise `poly_coeffs`.
  2. For each `n in 0..degree-1`, compute
     `phase = arccos(c_n / c_0)` (clipped to `[-1, 1]`).
  3. Return `[phase_0, ..., phase_{degree-1}]`.

### `approximate_threshold(threshold, degree, target_loss=0.5, max_loss=1.0) → list[float]`

Module-level convenience function. Dispatch:

- `degree ≤ 256` → uses `QSVTApproximator.approximate_threshold`
  (which calls `pyqsp` under the hood).
- `degree > 256` → uses `ChebyshevApproximator` directly.

This dispatch logic is the implementation of architecture
decision D4 ("pyqsp 0.2.0 with Chebyshev fallback for
high degree").

### Notes

- The hand-rolled Chebyshev path is **not a faithful QSP→QSVT
  conversion**; it is a first-order analytical approximation
  suitable for high-degree smoke tests but not for publication
  results. The production path is `pyqsp`.
- The `target_loss` and `max_loss` arguments are accepted for
  interface symmetry with `ThresholdFunction` but the current
  implementation does not use them; the threshold polynomial is
  parameterised only on `threshold` and `degree`.

---

## `threshold.py`

**Class** `ThresholdFunction`.

Constructs a Chebyshev-polynomial approximation of the **even
threshold function** that QSVT applies to the loss axis:

```
f(x) = 1   if x > threshold
       0.5 if x = threshold
       0   if x < threshold
```

### Constructor

```python
ThresholdFunction(
    threshold: float = 0.5,
    degree: int = 16,
    target_loss: float = 0.5,
    max_loss: float = 1.0,
)
```

- `threshold` ∈ `(0, 1)`.
- `degree` ∈ `[1, 2048]`.

### `polynomial_coefficients() → np.ndarray[degree+1]`

Chebyshev coefficients via discrete cosine transform. The
implementation:

1. Compute Chebyshev nodes `cos(π · k / degree)` for `k = 0..degree`.
2. Evaluate the step function at the nodes (binary
   `nodes > threshold`).
3. For each `n = 0..degree`, compute
   `c_n = (2/degree) · w_n · sum_j values[j] · T_n(nodes[j])`
   (with `w_0 = w_degree = 0.5`, else `1.0`).
4. Zero out all odd coefficients (to enforce even parity — the
   QSVT step function is even).

### `evaluate(x) → np.ndarray`

Evaluate the Chebyshev polynomial at points `x ∈ [-1, 1]`. Uses
the standard recurrence `T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)`.
The result is clipped to `[0, 1]`.

### `qsvt_phases() → list[float]`

Convert the polynomial coefficients to a phase list:

- `degree ≤ 256` → use `pyqsp.angle_sequence.QuantumSignalProcessingPhases`
  with the Chebyshev basis.
- `degree > 256` → fall back to the hand-rolled Chebyshev
  sequence via `_chebyshev_phases`.

Returns a list of length `degree`.

### `_pyqsp_phases(coeffs) → list[float]`

Calls `QuantumSignalProcessingPhases(poly, signal_operator="Wx",
method="sym_qsp", measurement="x", chebyshev_basis=True)`. The
`pyqsp` return type is `tuple | ndarray` depending on version;
this method normalises both to `list[float]`.

### `_chebyshev_phases(coeffs) → list[float]`

Hand-rolled analytical Chebyshev phase formula:
`phase_n = arccos(c_n / c_0)` for `n = 0..degree-1`. Used as
the fallback when `pyqsp` fails at high degree.

### Notes

- The degree-`d` Chebyshev approximation has `d+1` coefficients
  but only `d` phases (the leading constant is absorbed into
  the amplitude).
- For `d ≤ 16` the approximation is fairly coarse; the paper's
  experiments use `d ∈ {4, 8, 16}` for K=17 hardware-feasibility
  reasons (noisy hardware limits polynomial depth).

### Usage example

```python
from qsvt.threshold import ThresholdFunction

tf = ThresholdFunction(threshold=0.5, degree=16)
phases = tf.qsvt_phases()        # list of 16 floats
print(len(phases))                 # 16
print(phases[:3])                  # [phi_0, phi_1, phi_2]
```
