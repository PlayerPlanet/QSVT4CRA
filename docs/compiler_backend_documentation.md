# `compiler_backend/` — IBM Heron r3 hardware compilation

> The hardware-aware compiler for IBM Heron r3 backends
> (default `ibm_boston`). Pure Python helpers plus IBM Runtime
> integration. Designed so unit tests can run without an IBM
> token.

## Files

| File | Public API | Purpose |
|---|---|---|
| [`compiler_backend/heron.py`](#heronpy) | `CompileReport`, `HeronCompileConfig`, `add_measurements`, `compile_for_backend`, `load_ibm_backend`, `load_ibm_service`, `load_ibm_token`, `select_calibration_aware_layout` | IBM Heron compiler + token loading |

The package exposes:

```python
from compiler_backend import (
    CompileReport, DEFAULT_TOKEN_FILE, HeronCompileConfig,
    add_measurements, compile_for_backend, load_ibm_backend,
    load_ibm_service, load_ibm_token, select_calibration_aware_layout,
)
```

---

## `heron.py`

The single module in this package. ~540 lines.

### Constants

```python
TWO_QUBIT_OPS = ("ecr", "cx", "cz", "iswap", "rxx", "ryy", "rzz")
HERON_PREFERRED_OPS = ("ecr", "cz", "cx")
DEFAULT_TOKEN_FILE = ".ibm_token"
TOKEN_FILE_ENV = "IBM_API_KEY_FILE"
```

`TWO_QUBIT_OPS` is used for counting 2-qubit gates.
`HERON_PREFERRED_OPS` is the set of operations whose error
rates are queried from the backend target to drive the
calibration-aware layout selector.

### `HeronCompileConfig` (frozen dataclass)

Configuration for the calibration-aware Heron compiler:

```python
@dataclass(frozen=True)
class HeronCompileConfig:
    backend_name: str = "ibm_boston"
    optimization_level: int = 3
    seed_transpiler: int = 42
    calibration_aware_layout: bool = True
    api_key_env: str = "IBM_API_KEY"
    channel: str | None = None
    use_pyzx: bool = False
    pyzx_levels: tuple[str, ...] = ("basic",)
```

`use_pyzx=False` by default. On small/medium circuits the
`pyzx.basic_optimization` round-trip cuts depth ~10-40%, but
on the full K=17 degree-4 QSVT the re-translation step adds
routing overhead that *increases* depth ~30% on Heron's
heavy-hex. Set `use_pyzx=True` to opt in for benchmarks on
small circuits.

### `CompileReport` (frozen dataclass)

Summary of a hardware-aware compilation result:

```python
@dataclass(frozen=True)
class CompileReport:
    backend_name: str
    processor_type: str | None
    n_qubits: int
    depth: int
    two_qubit_depth_proxy: int
    gate_counts: dict[str, int]
    selected_physical_qubits: list[int]
    estimated_success_probability: float | None
    notes: str = ""
```

`as_dict()` returns a plain dict (used for JSON
serialisation).

### Token resolution

```python
load_ibm_token(
    api_key_env: str = "IBM_API_KEY",
    token_file: str | os.PathLike[str] | None = DEFAULT_TOKEN_FILE,
) → str
```

Resolves the IBM Quantum token from, in order:

1. `$IBM_API_KEY` (default; recommended for HPC / Slurm).
2. `$IBM_API_KEY_FILE` (path to a file whose first non-empty,
   non-comment line is the token).
3. `token_file` (default `.ibm_token` in cwd).

Raises `RuntimeError` with a clear message if none of the
three are available. **The token is never logged.**

### IBM Runtime helpers

```python
load_ibm_service(api_key_env="IBM_API_KEY", channel=None, token_file=DEFAULT_TOKEN_FILE) → QiskitRuntimeService
load_ibm_backend(backend_name="ibm_boston", api_key_env="IBM_API_KEY", channel=None, token_file=DEFAULT_TOKEN_FILE) → Backend
```

- `load_ibm_service` calls `load_ibm_token` then tries
  `QiskitRuntimeService(channel=candidate, token=token)` for
  each candidate channel. Tries `["ibm_quantum_platform",
  "ibm_cloud"]` if `channel is None`. Falls back to
  `QiskitRuntimeService(token=token)` for older runtimes
  (only if `channel is None`).
- `load_ibm_backend` calls `load_ibm_service` and then
  `service.backend(backend_name)` (with a
  `service.get_backend(backend_name)` fallback for old
  runtimes).

Both raise `RuntimeError` with the *most relevant* error
if all candidates fail (the last explicit-channel error, not
a confusing "channel required" message).

### `add_measurements(circuit, qubits=None, creg_name="meas") → QuantumCircuit`

Returns a copy of `circuit` with measurements on `qubits`
(default: all qubits). Existing classical bits are
preserved; a new `ClassicalRegister` is appended.

Raises `ValueError` if `qubits` is an empty list.

### Pure helpers (no IBM token required)

The following helpers are pure-Python and work without an
IBM token. They are extensively unit-tested.

#### `_backend_name(backend) → str`

Resolves the backend's name attribute (which may be a method
on older Qiskit versions).

#### `_processor_type(backend) → str | None`

Extracts the processor family and revision from
`backend.processor_type` (a dict with `family` and
`revision` keys). Returns `None` if unavailable.

#### `_target(backend) → Any | None`

Returns `backend.target`, or `None` if missing.

#### `_num_backend_qubits(backend) → int`

Reads `target.num_qubits` if available, else
`backend.configuration().n_qubits`, else 0.

#### `_iter_coupling_edges(backend) → list[tuple[int, int]]`

Iterates over the backend's coupling edges. Tries
`target.build_coupling_map()`, then `backend.coupling_map`,
then `backend.configuration().coupling_map`.

#### `_instruction_error_from_target(target, op_name, qargs) → float | None`

Reads the error rate for `(op_name, qargs)` from the target.
Returns `None` if not available.

#### `_gate_error(backend, qargs, op_names) → float | None`

Iterates over `op_names` and tries to find an error rate for
the `(op, qargs)` pair. If `qargs` is a 2-tuple, also tries
the reversed order (edge direction is sometimes
undirected in the target).

#### `_readout_error(backend, qubit) → float`

Reads the measurement error for the given qubit. Falls back
to `0.01` (1%) if no error rate is available.

#### `_edge_weight(backend, edge) → float`

Convenience: returns `_gate_error(backend, edge, HERON_PREFERRED_OPS)`
or 0.01 if `None`.

### `select_calibration_aware_layout(backend, n_qubits) → list[int]`

Greedy calibration-aware initial-layout heuristic for heavy-hex
Heron devices:

1. Start from the qubit with the lowest `incident_score`
   (`readout_error + min(incident_edge_gate_error)`).
2. Grow a connected patch by repeatedly adding the
   lowest-error boundary qubit.
3. Fall back to globally best unused qubit if the patch
   becomes disconnected.

Deterministic, well-suited for small QSVT smoke circuits.
Raises `ValueError` if `n_qubits > backend.num_qubits` or if
the patch cannot be grown to `n_qubits` (disconnected
target).

### `_two_qubit_count(gate_counts) → int`

Sum of `count` for `name in TWO_QUBIT_OPS`. Used as the
`two_qubit_depth_proxy` in `CompileReport`.

### `_success_probability_proxy(backend, gate_counts, selected_qubits) → float | None`

Crude estimate:

```
(1 - twoq_err_avg) ^ twoq_count  ·  (1 - readout_err_avg) ^ n_qubits
```

Returns `None` on overflow. Used to set
`estimated_success_probability` in `CompileReport`.

### `_try_pyzx_optimize(circuit, backend=None, levels=("basic",), seed_transpiler=42, optimization_level=3) → (optimized, notes)`

Round-trip the circuit through `pyzx.basic_optimization`
(or `pyzx.full_optimize` for Clifford+T circuits) and
re-translate to the backend's native basis. Returns
`(optimized, None)` on success or `(circuit, "error-msg")` on
failure.

**Note**: the pyzx round-trip leaves the circuit in
`{h, cx, cz, rz, x}`. When `backend` is provided, the
function re-translates via Qiskit's preset pass manager to
Heron's native `{ecr, rz, sx}`.

### `compile_for_backend(circuit, backend, config=None) → (QuantumCircuit, CompileReport)`

The main entry point. Steps:

1. If `config.calibration_aware_layout`, compute
   `selected_physical_qubits` via
   `select_calibration_aware_layout(backend, circuit.num_qubits)`.
2. Build a Qiskit preset pass manager with
   `initial_layout=selected` and run it on the circuit.
3. If `config.use_pyzx`, run `_try_pyzx_optimize` on the
   compiled circuit.
4. Build a `CompileReport` with the gate counts, depth,
   selected qubits, and the success-probability proxy.

Returns `(compiled, report)`. The `notes` field is a
semicolon-separated string of fallback events
(e.g., `"pyzx:42->31;layout-fallback:..."`).

### Usage example

```python
from qiskit import QuantumCircuit
from compiler_backend.heron import (
    HeronCompileConfig, compile_for_backend, load_ibm_token,
)

# Pure-Python helpers work without an IBM token
from compiler_backend.heron import select_calibration_aware_layout, add_measurements

qc = QuantumCircuit(3)
qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
measured = add_measurements(qc, qubits=[0, 2])

# To compile for a real IBM backend, you need the token:
# token = load_ibm_token()  # raises if not set
# backend = load_ibm_backend("ibm_boston")
# config = HeronCompileConfig(backend_name="ibm_boston", use_pyzx=True)
# compiled, report = compile_for_backend(measured, backend, config=config)
# print(report.depth, report.two_qubit_depth_proxy)
```
