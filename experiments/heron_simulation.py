"""Compile and simulate the K=17 Finnish mortgage QSVT circuit on Heron data.

Default workflow:

1. Build the pulled `Code.dataset_regions` toy mortgage dataset.
2. Build the low-degree expected-probability QSVT smoke circuit.
3. Fetch `ibm_boston` using `$IBM_API_KEY`.
4. Compile with calibration-aware Heron r3 layout.
5. Run a noisy Aer simulation seeded from the backend calibration model.

This script does **not** submit to the real QPU.  It is intended for LUMI/Aer
simulation after retrieving IBM backend target/calibration data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from compiler_backend.heron import (
    HeronCompileConfig,
    add_measurements,
    compile_for_backend,
    load_ibm_backend,
)


def build_finnish_mortgage_qsvt_circuit(
    degree: int = 4,
    target_loss_fraction: float = 0.5,
):
    """Build the K=17 Finnish mortgage expected-probability smoke circuit."""

    from Code import dataset_regions as ds
    from Code.circuitsCRA import get_expected_probability_circuit
    from Code.multivariateGCI import MultivariateGCI_Linear

    uncertainty_model = MultivariateGCI_Linear(
        n_normal=ds.n_z,
        normal_max_value=ds.z_max,
        p_zeros=ds.p_zeros,
        rhos=ds.rhos,
        F_list=ds.F_values,
    )
    max_loss = float(np.sum(ds.lgd))
    target_loss = target_loss_fraction * max_loss

    # Degree-smoke phases.  The phase-quality/accuracy sweep remains separate;
    # this hardware workflow first verifies compilation, layout, and noisy-run
    # feasibility on a small QSVT instance.
    phases = [0.0] * max(2, int(degree))
    circuit, objective, ok = get_expected_probability_circuit(
        K=ds.K,
        uncertainity_model=uncertainty_model,
        lgd=list(ds.lgd),
        target_loss=target_loss,
        phases=phases,
        threshold=0.5,
    )
    if not ok:
        raise RuntimeError("get_expected_probability_circuit returned ok=False")

    # Register order from Code.circuitsCRA:
    #   z factors | QSVT State(K) | QSVT Target(1) | QSVT aux(1)
    target_qubit_index = circuit.num_qubits - 2
    return circuit, {
        "K": ds.K,
        "regions": list(ds.regions),
        "sum_lgd": max_loss,
        "target_loss": target_loss,
        "target_loss_fraction": target_loss_fraction,
        "target_qubit_index": target_qubit_index,
        "objective_qubits": objective.num_qubits,
    }


def _jsonable_counts(counts: Any) -> dict[str, int]:
    return {str(k): int(v) for k, v in dict(counts).items()}


def run_heron_simulation(
    backend_name: str,
    degree: int,
    shots: int,
    target_loss_fraction: float,
    output: str,
    api_key_env: str,
    channel: str | None,
    optimization_level: int,
    seed_transpiler: int,
    measure_all: bool,
    noisy: bool,
) -> dict[str, Any]:
    t0 = time.time()
    circuit, dataset_meta = build_finnish_mortgage_qsvt_circuit(
        degree=degree,
        target_loss_fraction=target_loss_fraction,
    )

    measured_qubits = None if measure_all else [int(dataset_meta["target_qubit_index"])]
    measured = add_measurements(circuit, qubits=measured_qubits)

    backend = load_ibm_backend(
        backend_name=backend_name,
        api_key_env=api_key_env,
        channel=channel,
    )
    config = HeronCompileConfig(
        backend_name=backend_name,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        calibration_aware_layout=True,
        api_key_env=api_key_env,
        channel=channel,
    )
    compiled, report = compile_for_backend(measured, backend, config=config)

    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("qiskit-aer is required for Heron simulation") from exc

    if noisy:
        try:
            simulator = AerSimulator.from_backend(backend)
        except Exception:
            simulator = AerSimulator()
    else:
        simulator = AerSimulator()

    job = simulator.run(compiled, shots=shots, seed_simulator=seed_transpiler)
    result = job.result()
    counts = _jsonable_counts(result.get_counts())
    total = sum(counts.values()) or 1
    prob_one = sum(v for k, v in counts.items() if k.replace(" ", "")[-1] == "1") / total

    payload = {
        "backend_name": backend_name,
        "degree": degree,
        "shots": shots,
        "noisy": noisy,
        "measure_all": measure_all,
        "dataset": dataset_meta,
        "precompile": {
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "count_ops": {str(k): int(v) for k, v in circuit.count_ops().items()},
        },
        "compile_report": report.as_dict(),
        "counts": counts,
        "target_prob_one": float(prob_one),
        "runtime_seconds": time.time() - t0,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile and Aer-simulate the Finnish mortgage QSVT circuit on Heron calibration data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend-name", default="ibm_boston")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--target-loss-fraction", type=float, default=0.5)
    parser.add_argument("--output", default="results/heron_k17_simulation.json")
    parser.add_argument("--api-key-env", default="IBM_API_KEY")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument("--measure-all", action="store_true")
    parser.add_argument("--noiseless", action="store_true", help="Use ideal Aer instead of AerSimulator.from_backend")
    args = parser.parse_args(argv)

    payload = run_heron_simulation(
        backend_name=args.backend_name,
        degree=args.degree,
        shots=args.shots,
        target_loss_fraction=args.target_loss_fraction,
        output=args.output,
        api_key_env=args.api_key_env,
        channel=args.channel,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
        measure_all=args.measure_all,
        noisy=not args.noiseless,
    )
    print(json.dumps({
        "output": args.output,
        "backend_name": payload["backend_name"],
        "compiled_depth": payload["compile_report"]["depth"],
        "compiled_two_qubit_proxy": payload["compile_report"]["two_qubit_depth_proxy"],
        "target_prob_one": payload["target_prob_one"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
