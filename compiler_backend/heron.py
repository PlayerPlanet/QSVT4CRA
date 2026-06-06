"""IBM Heron compiler backend for QSVT4CRA.

The functions in this module are intentionally split into pure helpers and
IBM Runtime helpers.  Tests can exercise the pure helpers without a token,
while production/LUMI runs can fetch ``ibm_boston`` through
``$IBM_API_KEY`` and compile against its current calibration/target data.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Iterable, Mapping, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister


TWO_QUBIT_OPS = ("ecr", "cx", "cz", "iswap", "rxx", "ryy", "rzz")
HERON_PREFERRED_OPS = ("ecr", "cz", "cx")


@dataclass(frozen=True)
class HeronCompileConfig:
    """Configuration for calibration-aware Heron compilation."""

    backend_name: str = "ibm_boston"
    optimization_level: int = 3
    seed_transpiler: int = 42
    calibration_aware_layout: bool = True
    api_key_env: str = "IBM_API_KEY"
    channel: str | None = None


@dataclass(frozen=True)
class CompileReport:
    """Summary of a hardware-aware compilation result."""

    backend_name: str
    processor_type: str | None
    n_qubits: int
    depth: int
    two_qubit_depth_proxy: int
    gate_counts: dict[str, int]
    selected_physical_qubits: list[int]
    estimated_success_probability: float | None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ibm_service(api_key_env: str = "IBM_API_KEY", channel: str | None = None):
    """Create a Qiskit Runtime service from an environment variable.

    The token is never printed or returned.  We try both current and older IBM
    channel names because the project has been used across Qiskit Runtime
    versions.
    """

    token = os.environ.get(api_key_env)
    if not token:
        raise RuntimeError(
            f"Missing IBM Quantum token: set ${api_key_env} before fetching backend calibration data."
        )

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "qiskit-ibm-runtime is required for IBM backend access. "
            "Install it or run the compiler in --offline mode."
        ) from exc

    channels = [channel] if channel else ["ibm_quantum_platform", "ibm_quantum"]
    last_error: Exception | None = None
    for candidate in channels:
        try:
            return QiskitRuntimeService(channel=candidate, token=token)
        except Exception as exc:  # pragma: no cover - provider/version dependent
            last_error = exc

    # Older runtime versions sometimes infer the channel from account metadata.
    try:
        return QiskitRuntimeService(token=token)
    except Exception as exc:  # pragma: no cover - provider/version dependent
        if last_error is not None:
            raise RuntimeError(f"Could not initialize IBM Runtime service: {last_error}") from exc
        raise


def load_ibm_backend(
    backend_name: str = "ibm_boston",
    api_key_env: str = "IBM_API_KEY",
    channel: str | None = None,
):
    """Load an IBM backend, defaulting to Heron r3 ``ibm_boston``."""

    service = load_ibm_service(api_key_env=api_key_env, channel=channel)
    if hasattr(service, "backend"):
        return service.backend(backend_name)
    return service.get_backend(backend_name)  # pragma: no cover - old runtime


def add_measurements(
    circuit: QuantumCircuit,
    qubits: Sequence[int] | None = None,
    creg_name: str = "meas",
) -> QuantumCircuit:
    """Return a copy of ``circuit`` with measurements on ``qubits``.

    If ``qubits`` is ``None``, all qubits are measured.  Existing classical
    bits are preserved and a new register is appended.
    """

    measured = circuit.copy()
    if qubits is None:
        qubits = list(range(measured.num_qubits))
    qubits = list(qubits)
    if not qubits:
        raise ValueError("At least one qubit must be measured.")

    creg = ClassicalRegister(len(qubits), creg_name)
    measured.add_register(creg)
    for cidx, qidx in enumerate(qubits):
        measured.measure(qidx, creg[cidx])
    return measured


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    return str(name or "unknown")


def _processor_type(backend: Any) -> str | None:
    props = getattr(backend, "processor_type", None)
    if isinstance(props, Mapping):
        family = props.get("family")
        revision = props.get("revision")
        return " ".join(str(x) for x in (family, revision) if x)
    return None


def _target(backend: Any) -> Any | None:
    return getattr(backend, "target", None)


def _num_backend_qubits(backend: Any) -> int:
    target = _target(backend)
    if target is not None and getattr(target, "num_qubits", None):
        return int(target.num_qubits)
    config = getattr(backend, "configuration", lambda: None)()
    if config is not None and getattr(config, "n_qubits", None):
        return int(config.n_qubits)
    return 0


def _iter_coupling_edges(backend: Any) -> list[tuple[int, int]]:
    target = _target(backend)
    if target is not None:
        try:
            cmap = target.build_coupling_map()
            if cmap is not None:
                return [(int(a), int(b)) for a, b in cmap.get_edges()]
        except Exception:
            pass
    cmap = getattr(backend, "coupling_map", None)
    if cmap is not None:
        try:
            return [(int(a), int(b)) for a, b in cmap.get_edges()]
        except Exception:
            return [(int(a), int(b)) for a, b in cmap]
    config = getattr(backend, "configuration", lambda: None)()
    raw = getattr(config, "coupling_map", None) if config is not None else None
    return [(int(a), int(b)) for a, b in (raw or [])]


def _instruction_error_from_target(target: Any, op_name: str, qargs: tuple[int, ...]) -> float | None:
    try:
        inst_map = target[op_name]
        props = inst_map.get(qargs)
        if props is not None and getattr(props, "error", None) is not None:
            return float(props.error)
    except Exception:
        return None
    return None


def _gate_error(backend: Any, qargs: tuple[int, ...], op_names: Sequence[str]) -> float | None:
    target = _target(backend)
    if target is not None:
        for op_name in op_names:
            err = _instruction_error_from_target(target, op_name, qargs)
            if err is None and len(qargs) == 2:
                err = _instruction_error_from_target(target, op_name, (qargs[1], qargs[0]))
            if err is not None:
                return err

    props = getattr(backend, "properties", lambda: None)()
    if props is not None:
        for op_name in op_names:
            try:
                return float(props.gate_error(op_name, list(qargs)))
            except Exception:
                if len(qargs) == 2:
                    try:
                        return float(props.gate_error(op_name, [qargs[1], qargs[0]]))
                    except Exception:
                        pass
    return None


def _readout_error(backend: Any, qubit: int) -> float:
    target = _target(backend)
    if target is not None:
        err = _instruction_error_from_target(target, "measure", (qubit,))
        if err is not None:
            return err
    props = getattr(backend, "properties", lambda: None)()
    if props is not None:
        try:
            return float(props.readout_error(qubit))
        except Exception:
            pass
    return 0.01


def _edge_weight(backend: Any, edge: tuple[int, int]) -> float:
    err = _gate_error(backend, edge, HERON_PREFERRED_OPS)
    return float(err if err is not None else 0.01)


def select_calibration_aware_layout(backend: Any, n_qubits: int) -> list[int]:
    """Select a connected low-error physical qubit subset.

    This is a greedy calibration-aware initial-layout heuristic: start from the
    best readout/incident-edge qubit, then grow a connected patch by repeatedly
    adding the lowest-error boundary qubit.  It is deterministic and works well
    for small QSVT smoke circuits on heavy-hex Heron devices.
    """

    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    n_backend = _num_backend_qubits(backend)
    if n_backend and n_qubits > n_backend:
        raise ValueError(f"Circuit needs {n_qubits} qubits, backend has {n_backend}.")

    edges = _iter_coupling_edges(backend)
    if not edges:
        return list(range(n_qubits))

    adjacency: dict[int, set[int]] = {}
    weights: dict[tuple[int, int], float] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
        w = _edge_weight(backend, (a, b))
        weights[(a, b)] = w
        weights[(b, a)] = w

    def incident_score(q: int) -> float:
        incident = [weights[(q, nb)] for nb in adjacency.get(q, ())]
        best_edge = min(incident) if incident else 0.05
        return _readout_error(backend, q) + best_edge

    start = min(adjacency, key=incident_score)
    selected = [start]
    selected_set = {start}

    while len(selected) < n_qubits:
        boundary: list[tuple[float, int]] = []
        for q in selected:
            for nb in adjacency.get(q, ()):
                if nb in selected_set:
                    continue
                score = weights[(q, nb)] + _readout_error(backend, nb)
                boundary.append((score, nb))
        if not boundary:
            # Disconnected target fallback: add globally best unused qubit.
            unused = [q for q in range(n_backend or max(adjacency) + 1) if q not in selected_set]
            if not unused:
                break
            nb = min(unused, key=incident_score)
        else:
            _, nb = min(boundary, key=lambda item: (item[0], item[1]))
        selected.append(nb)
        selected_set.add(nb)

    if len(selected) < n_qubits:
        raise ValueError(f"Could not select {n_qubits} physical qubits from backend topology.")
    return selected


def _two_qubit_count(gate_counts: Mapping[str, int]) -> int:
    return int(sum(count for name, count in gate_counts.items() if name in TWO_QUBIT_OPS))


def _success_probability_proxy(
    backend: Any,
    gate_counts: Mapping[str, int],
    selected_qubits: Sequence[int],
) -> float | None:
    if not selected_qubits:
        return None
    readout = sum(_readout_error(backend, q) for q in selected_qubits) / len(selected_qubits)
    twoq_errs = []
    for a, b in _iter_coupling_edges(backend):
        if a in selected_qubits and b in selected_qubits:
            twoq_errs.append(_edge_weight(backend, (a, b)))
    twoq = sum(twoq_errs) / len(twoq_errs) if twoq_errs else 0.01
    twoq_count = _two_qubit_count(gate_counts)
    meas_count = max(1, len(selected_qubits))
    try:
        return float(prod([(1.0 - twoq) ** twoq_count, (1.0 - readout) ** meas_count]))
    except Exception:
        return None


def compile_for_backend(
    circuit: QuantumCircuit,
    backend: Any,
    config: HeronCompileConfig | None = None,
) -> tuple[QuantumCircuit, CompileReport]:
    """Compile ``circuit`` for a Heron backend using calibration-aware layout."""

    config = config or HeronCompileConfig(backend_name=_backend_name(backend))
    selected: list[int] = []
    initial_layout: list[int] | None = None
    notes: list[str] = []

    if config.calibration_aware_layout:
        try:
            selected = select_calibration_aware_layout(backend, circuit.num_qubits)
            initial_layout = selected
        except Exception as exc:
            notes.append(f"layout-fallback:{exc}")

    try:
        try:
            from qiskit.transpiler import generate_preset_pass_manager
        except ImportError:  # pragma: no cover - old qiskit path
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
            initial_layout=initial_layout,
        )
        compiled = pm.run(circuit)
    except Exception as exc:
        notes.append(f"preset-passmanager-fallback:{exc}")
        compiled = transpile(
            circuit,
            backend=backend,
            optimization_level=config.optimization_level,
            seed_transpiler=config.seed_transpiler,
            initial_layout=initial_layout,
        )

    counts = {str(k): int(v) for k, v in compiled.count_ops().items()}
    if not selected:
        selected = list(range(min(circuit.num_qubits, _num_backend_qubits(backend) or circuit.num_qubits)))

    report = CompileReport(
        backend_name=_backend_name(backend),
        processor_type=_processor_type(backend),
        n_qubits=compiled.num_qubits,
        depth=int(compiled.depth() or 0),
        two_qubit_depth_proxy=_two_qubit_count(counts),
        gate_counts=counts,
        selected_physical_qubits=[int(q) for q in selected],
        estimated_success_probability=_success_probability_proxy(backend, counts, selected),
        notes=";".join(notes),
    )
    return compiled, report
