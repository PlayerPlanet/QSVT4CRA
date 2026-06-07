# `loader/` — Quantum state-preparation loaders

> The `loader/` package turns a posterior sample `θ` into a
> Qiskit `QuantumCircuit` that encodes the corresponding loss
> distribution. It is the drop-in replacement for the
> `MultivariateGCI_*` loaders in `Code/`, scaling as `K + 1`
> qubits (not `2^K`).

## Files

| File | Class | Role |
|---|---|---|
| [`loader/posterior_factor_copula.py`](#posterior-factor-copulapy) | `PosteriorFactorCopulaLoader` | The drop-in replacement for `MultivariateGCI_*` |
| [`loader/amplitude_loader.py`](#amplitude-loaderpy) | `AmplitudeLoader` | Low-level amplitude-encoding primitive |

The package exposes:

```python
from loader import PosteriorFactorCopulaLoader, AmplitudeLoader
```

---

## `posterior_factor_copula.py`

**Class** `PosteriorFactorCopulaLoader(QuantumCircuit)`.

This is the implementation of architecture decision **D3** — the
new loader that replaces `MultivariateGCI_Linear` /
`MultivariateGCI_Poly` with an amplitude-encoding of the
factor-copula loss distribution.

### Constructor

```python
PosteriorFactorCopulaLoader(
    theta: np.ndarray,
    K: int = 10,
    max_loss: float = 1.0,
    name: str = "PFC",
)
```

- `theta`: a posterior sample, layout
  `[:K] = factor_loadings, [K:2K] = p_zeros, [2K] = tail_dep,
   [2K+1] = rho, [2K+2] = nu, [2K+3] = spare`.
- `K`: number of loans.
- `max_loss`: maximum portfolio loss for normalisation.
- Internally sets `lgd = [0.40] * K` (fixed at the apartment-loan
  LGD midpoint). The per-region `Code.dataset_regions.lgd`
  values are **not** used by this loader — the loader targets
  the unit-LGD scenario for swapability with
  `get_expected_probability_circuit`.

### `num_qubits` property

`K + 1` (K state + 1 target). Compare to
`MultivariateGCI_Linear` which uses `n_normal · sectors + K`
qubits, i.e. **exponential** in the number of Gaussian sectors.

### `_build_circuit()`

1. Instantiates a `GaussianFactorCopula(K=K, seed=42)` and
   draws `n_samples=10_000` uniform marginals `U`.
2. Computes scenario probabilities via
   `_compute_scenario_probs(U, K)`:
   - Convert `U` to defaults via `U < p_zeros[None, :]`.
   - Encode default patterns as integers via dot-product with
     `2 ** arange(K)`.
   - Histogram (`np.bincount`) over the `2^K` scenarios, normalise
     to probabilities.
3. Computes per-scenario losses via
   `_mapping(j, lgd)` (sums the LGDs of the loans whose bit is
   set).
4. Normalises losses to `[0, 1]` via `/ max_loss`.
5. Builds an `AmplitudeLoader(K, amplitudes)` and appends it
   to the circuit. The amplitudes are `sqrt(clip(probs, 1e-10, 1))`
   then L2-normalised, so the quantum state on the K state
   qubits is `Σ_j sqrt(P(scenario j)) |j⟩`.

### Properties

- `num_qubits` — total qubits.
- `K` — number of state qubits (loans).
- `lgd` — copy of the LGD array.

### Caveat

`_compute_scenario_probs` is **stochastic**: it relies on MC
samples from the copula, not the analytical posterior. For
`n_samples=10_000` the scenario-probability histogram is fairly
coarse; for accurate downstream risk metrics, increase
`n_samples` (or compute the analytical histogram from the
copula). This is the same caveat that
`QSVTRiskCircuit._build_circuit` shares.

---

## `amplitude_loader.py`

**Class** `AmplitudeLoader(QuantumCircuit)`.

Low-level amplitude-encoding primitive. Maps basis states
`|j⟩|0⟩ ↦ |j⟩(cos(α_j)|0⟩ + sin(α_j)|1⟩)` for a vector of
values `v_j` of length `2^K`, where
`α_j = arcsin(v_j / max(|v|))`.

### Constructor

```python
AmplitudeLoader(
    num_state_qubits: int,
    values: np.ndarray,
    name: str = "amp_load",
)
```

- `num_state_qubits`: K (so the state register holds `2^K`
  values).
- `values`: array of length `2^K` (must match).
- Internal: L2-normalises `values` to get the amplitudes, then
  scales to `[-1, 1]` and computes `arcsin` to get the rotation
  angles. The angles are doubled (Qiskit's `CRY` convention).

### `_build_circuit()`

Currently **places CRY gates in a Schmidt-like cascade**
(stride pattern). The decomposition is heuristic; for
production use, prefer `Code.AmplitudeLoading.AmplitudeLoadingVar`
which has a tested Schmidt decomposition.

### Properties

- `num_ancillas` — 0 (no ancilla qubits used by this loader).
- `to_gate()` — wraps the circuit in a `Gate` for reusability.
- `_decompose()` — returns the circuit decomposed to basis gates.

### Known issue

The `AmplitudeLoader` in this package is the QSVT4CRA
research-run implementation, distinct from
`Code.AmplitudeLoading.AmplitudeLoadingVar`. They are
**not** interchangeable: the latter is a tested class used in
production, the former is the research variant. In
[`qsvt/circuit.py`](qsvt_documentation.md), the production
path uses `Code.AmplitudeLoading.AmplitudeLoadingVar` directly;
`AmplitudeLoader` is only used by
`PosteriorFactorCopulaLoader`.

### Usage example

```python
import numpy as np
from loader.amplitude_loader import AmplitudeLoader

K = 4
values = np.array([0.1, 0.5, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6])
loader = AmplitudeLoader(K, values)
print(loader.num_qubits)   # 5 (K state + 1 target)
print(loader.num_ancillas)  # 0
```
