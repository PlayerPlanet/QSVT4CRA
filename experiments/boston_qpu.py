"""Compile, submit, and benchmark the K=17 Finnish mortgage QSVT circuit on
the real IBM ``ibm_boston`` (Heron r3) QPU, then score the result against
classical VaR/CVaR ground truth sampled from the same Gaussian-Conditional-
Independence (GCI) model that the circuit encodes.

Pipeline
--------

1. Build the expected-probability QSVT circuit with **real Chebyshev
   threshold phases** computed by ``qsvt.approximator.approximate_threshold``
   (via ``pyqsp``).  The all-zero "smoke" phases used by
   ``experiments/heron_simulation.py`` are explicitly avoided — they make
   the QSVT projector the identity and the answer bit is not a threshold
   probability at all.
2. Compute a classical Monte Carlo reference for the same loss distribution
   using the GCI parameterisation from ``Code.multivariateGCI``: latent
   ``Z ~ N(0, I)``, per-region default probability
   ``PD_i(z) = Phi((Phi^{-1}(p_zero_i) - rho * F_i . z) / sqrt(1-rho))``,
   independent Bernoulli defaults, ``loss = sum_i X_i * lgd_i``.
3. Fetch ``ibm_boston`` and compile the circuit with the same
   calibration-aware Heron pass manager used by ``heron_simulation.py``.
4. The QSVT answer lives on the **AUX qubit** (``num_qubits - 1``), not the
   Target qubit.  The Target only holds the encoded loss value via
   ``AmplitudeLoadingVar``; the AUX carries ``P(loss > target_loss)`` after
   the H-H sandwich around the QSP projector rotations.  This script
   measures the AUX.
5. Submit to the **real** QPU via ``qiskit_ibm_runtime.SamplerV2``, wait for
   the job, and recover the AUX marginal counts.
6. Optionally run a **per-qubit readout mitigation** round: two calibration
   circuits that prepare ``|0>`` and ``|1>`` on the AUX's physical qubit,
   build the 2x2 confusion matrix ``C`` and apply ``C^-1`` to the main
   counts.  Skipped with ``--no-mitigation``.
7. Compute ``abs/rel`` error between the mitigated ``P(AUX=1)`` and the
   classical ``P(loss > target_loss)`` (VaR/CVaR also reported) and persist
   the full payload (compile report, job IDs, counts, calibration, classical
   metrics, error dict) to ``--output``.

IBM access token
----------------

The token is resolved by ``compiler_backend.heron.load_ibm_token`` in this
order:

    1. ``$IBM_API_KEY``  (recommended for HPC / Slurm)
    2. ``$IBM_API_KEY_FILE``  (path to a token file)
    3. ``.ibm_token`` in the current working directory  (gitignored)

Set the token before running, e.g. ``export IBM_API_KEY=...``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from compiler_backend.heron import (
    DEFAULT_TOKEN_FILE,
    HeronCompileConfig,
    add_measurements,
    compile_for_backend,
    load_ibm_backend,
    load_ibm_service,
)


# ---------------------------------------------------------------------------
# Circuit construction with real QSVT phases
# ---------------------------------------------------------------------------


def build_finnish_mortgage_qsvt_circuit(
    degree: int = 4,
    target_loss_fraction: float = 0.5,
):
    """Build the K=17 Finnish mortgage expected-probability circuit with
    real Chebyshev threshold phases.

    Returns ``(circuit, meta)`` where ``meta`` includes the **AUX qubit
    index** (the QSVT projector answer bit) and the dataset-level scalars.
    """

    from Code import dataset_regions as ds
    from Code.circuitsCRA import get_expected_probability_circuit
    from Code.multivariateGCI import MultivariateGCI_Linear
    from qsvt.approximator import approximate_threshold

    uncertainty_model = MultivariateGCI_Linear(
        n_normal=ds.n_z,
        normal_max_value=ds.z_max,
        p_zeros=ds.p_zeros,
        rhos=ds.rhos,
        F_list=ds.F_values,
    )
    max_loss = float(np.sum(ds.lgd))
    target_loss = target_loss_fraction * max_loss

    # Real Chebyshev-QSP phases for the even threshold function with the same
    # normalized target/middle the rest of the codebase uses.
    degree = max(2, int(degree))
    phases = approximate_threshold(
        threshold=0.5,
        degree=degree,
        target_loss=0.5,
        max_loss=1.0,
    )

    circuit, _objective, ok = get_expected_probability_circuit(
        K=ds.K,
        uncertainity_model=uncertainty_model,
        lgd=list(ds.lgd),
        target_loss=target_loss,
        phases=phases,
        threshold=0.5,
    )
    if not ok:
        raise RuntimeError("get_expected_probability_circuit returned ok=False")

    # Register order from Code.circuitsCRA.get_expected_probability_circuit:
    #   z (n_z) | State (K) | Target (1) | aux_qubit (1)
    # The QSVT answer bit is the AUX qubit (last), not the Target.
    aux_qubit_index = circuit.num_qubits - 1
    target_qubit_index = circuit.num_qubits - 2
    meta = {
        "K": ds.K,
        "regions": list(ds.regions),
        "sum_lgd": max_loss,
        "target_loss": target_loss,
        "target_loss_fraction": target_loss_fraction,
        "target_qubit_index": target_qubit_index,
        "aux_qubit_index": aux_qubit_index,
        "phases": [float(p) for p in phases],
        "lgd": list(map(float, ds.lgd)),
        "p_zeros": list(map(float, ds.p_zeros)),
        "rhos": list(map(float, ds.rhos)),
        "F_values": [list(map(float, row)) for row in ds.F_values],
    }
    return circuit, meta


# ---------------------------------------------------------------------------
# Classical reference (GCI Monte Carlo)
# ---------------------------------------------------------------------------


def classical_gci_losses(
    p_zeros,
    rhos,
    f_values,
    lgd,
    n_scenarios: int = 200_000,
    n_z: int = 2,
    z_max: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample portfolio losses from the GCI model encoded by
    ``Code.multivariateGCI.MultivariateGCI_Linear``.

    The latent ``Z ~ N(0, I_{n_z})`` is optionally truncated to ``[-z_max,
    +z_max]`` to mirror the truncation imposed by the discrete quantum
    state preparation.  Per-region default probability follows the standard
    Gaussian conditional-independence parameterisation::

        PD_i(z) = Phi((Phi^{-1}(p_zero_i) - rho * F_i . z) / sqrt(1 - rho))

    Defaults are sampled independently given ``Z`` and the portfolio loss
    is ``sum_i X_i * lgd_i``.
    """

    rng = np.random.default_rng(seed)
    p_zeros = np.asarray(p_zeros, dtype=np.float64)
    rhos = np.asarray(rhos, dtype=np.float64)
    f_values = np.asarray(f_values, dtype=np.float64)
    lgd = np.asarray(lgd, dtype=np.float64)

    K = p_zeros.shape[0]
    assert rhos.shape == (K,)
    assert f_values.shape == (K, n_z), f_values.shape

    z = rng.standard_normal(size=(n_scenarios, n_z))
    if z_max is not None:
        z = np.clip(z, -z_max, z_max)

    psi = norm.ppf(p_zeros) / np.sqrt(1.0 - rhos)
    fz = z @ f_values.T
    pd = norm.cdf((psi[None, :] - rhos[None, :] * fz) / np.sqrt(1.0 - rhos[None, :]))
    pd = np.clip(pd, 0.0, 1.0)

    indicators = rng.random(size=pd.shape) < pd
    losses = indicators @ lgd
    return losses.astype(np.float64)


def classical_reference(
    meta: dict,
    n_scenarios: int = 200_000,
    alphas: tuple[float, ...] = (0.95, 0.99, 0.999),
    seed: int = 0,
) -> dict:
    """Compute the classical VaR/CVaR/tail-probability reference for the
    circuit's loss model.  ``tail_at_target_loss`` is the value the quantum
    circuit's tail estimate ``P(AUX=1)`` is benchmarked against.
    """

    from Code import dataset_regions as ds
    from metrics.var_cvar import var_cvar

    losses = classical_gci_losses(
        p_zeros=meta["p_zeros"],
        rhos=meta["rhos"],
        f_values=meta["F_values"],
        lgd=meta["lgd"],
        n_scenarios=n_scenarios,
        n_z=ds.n_z,
        z_max=ds.z_max,
        seed=seed,
    )

    metrics = var_cvar(losses, alphas=list(alphas))
    tail_at_target = float(np.mean(losses > meta["target_loss"]))
    metrics["tail_at_target_loss"] = tail_at_target
    metrics["n_scenarios"] = int(n_scenarios)
    metrics["max_loss"] = float(meta["sum_lgd"])
    return metrics


# ---------------------------------------------------------------------------
# QPU execution
# ---------------------------------------------------------------------------


def _extract_bit_counts(bit_array_or_data, bit_index: int = 0) -> dict[str, int]:
    """Pull the marginal bitstring counts for ``bit_index`` from a
    SamplerV2 primitive result.

    Accepts either a ``qiskit.primitives.containers.BitArray`` (when
    called on the inner result of a pre-1.0 / legacy path) or a
    ``DataBin`` (the SamplerV2 primitive result in qiskit 2.x).  The
    DataBin exposes ``creg_names`` and per-register attributes; we
    pick the first BitArray we find.

    For multi-bit registers we slice to ``bit_index`` (default 0 = the
    single measured bit in our calibration circuits).
    """

    # Resolve a BitArray from whatever the caller handed us
    bit_array = bit_array_or_data
    if not hasattr(bit_array, "get_int_counts"):
        # Likely a DataBin
        creg_names = list(getattr(bit_array, "creg_names", []) or [])
        if creg_names:
            bit_array = getattr(bit_array, creg_names[-1])
        else:
            data = getattr(bit_array, "_data", None) or {}
            if not data:
                return {"0": 0, "1": 0}
            bit_array = next(iter(data.values()))

    try:
        if getattr(bit_array, "num_bits", 1) > 1 and hasattr(bit_array, "slice_bits"):
            bit_array = bit_array.slice_bits(bit_index, bit_index + 1)
    except Exception:
        pass

    int_counts = bit_array.get_int_counts()
    counts = {"0": 0, "1": 0}
    for k, v in int_counts.items():
        key = "1" if int(k) & 1 else "0"
        counts[key] += int(v)
    return counts


def _mitigate_single_qubit(counts: dict[str, int], confusion: np.ndarray) -> dict[str, int]:
    """Apply per-qubit readout mitigation to a 1-bit ``{"0": n0, "1": n1}``
    counts dict using a 2x2 confusion matrix.

    The confusion matrix convention is ``C[i, j] = P(measure=i | state=j)``,
    so columns are the two "prep then measure" calibration columns and
    ``P_obs = C @ P_true`` for a single qubit.

    Returns a *new* counts dict with corrected n0/n1 (rounded to int, may be
    slightly non-integer before rounding) and a flag in the dict indicating
    if the matrix was ill-conditioned.
    """

    total = counts.get("0", 0) + counts.get("1", 0)
    if total <= 0:
        return {"0": 0, "1": 0, "_mitigated": True, "_condition_number": float("inf")}

    p_obs = np.array(
        [counts.get("0", 0) / total, counts.get("1", 0) / total], dtype=np.float64
    )
    try:
        c_inv = np.linalg.inv(confusion)
        cond = float(np.linalg.cond(confusion))
        p_true = c_inv @ p_obs
        p_true = np.clip(p_true, 0.0, 1.0)
        p_true = p_true / p_true.sum()  # renormalise
    except np.linalg.LinAlgError:
        return {"0": counts.get("0", 0), "1": counts.get("1", 0), "_mitigated": False}

    return {
        "0": int(round(float(p_true[0]) * total)),
        "1": int(round(float(p_true[1]) * total)),
        "_mitigated": True,
        "_condition_number": cond,
    }


def _build_calibration_circuits(aux_physical_qubit: int, cal_shots: int, backend):
    """Pre-transpile the ``|0>`` and ``|1>`` prep calibration circuits
    onto ``aux_physical_qubit`` of ``backend``.  Returns
    ``(cal_0_transpiled, cal_1_transpiled)``.
    """

    from qiskit import QuantumCircuit, transpile

    cal_0 = QuantumCircuit(1, 1, name="cal_prep_0")
    cal_0.measure(0, 0)
    cal_1 = QuantumCircuit(1, 1, name="cal_prep_1")
    cal_1.x(0)
    cal_1.measure(0, 0)

    cal_0_t = transpile(
        cal_0, backend=backend, optimization_level=1, initial_layout=[aux_physical_qubit]
    )
    cal_1_t = transpile(
        cal_1, backend=backend, optimization_level=1, initial_layout=[aux_physical_qubit]
    )
    return cal_0_t, cal_1_t


def submit_main_and_mitigate(
    compiled_circuit,
    backend,
    shots: int,
    aux_qubit_index: int,
    apply_mitigation: bool,
    cal_shots: int = 1024,
    report=None,
):
    """Submit the main experiment (and optionally the calibration circuits)
    to the real QPU in a **single** batched job, then return the AUX
    marginal counts and calibration metadata.

    Batching the main and calibration circuits as separate *pubs* of a
    single ``SamplerV2.run()`` call ensures all three are queued together
    and run back-to-back, which avoids the "stuck in queue" problem of
    submitting them sequentially and avoids any temporal drift between
    main and calibration (the readout error can change between jobs).
    """

    try:
        from qiskit_ibm_runtime import SamplerV2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "qiskit-ibm-runtime is required to submit to the real QPU."
        ) from exc

    service = load_ibm_service()

    pubs = [compiled_circuit]
    pub_shots = [shots]
    if apply_mitigation:
        aux_phys = _resolve_aux_physical_qubit(
            compiled_circuit, aux_qubit_index, report=report
        )
        cal_0_t, cal_1_t = _build_calibration_circuits(aux_phys, cal_shots, backend)
        pubs.extend([cal_0_t, cal_1_t])
        pub_shots.extend([cal_shots, cal_shots])

    print(
        f"  submitting batched job: 1 main ({shots} shots) + "
        f"{'2 cal (' + str(cal_shots) + ' shots each)' if apply_mitigation else 'no cal'}",
        flush=True,
    )
    sampler = SamplerV2(mode=backend, options={"default_shots": shots})
    # Per-pub shots via the (circuit, _, shots) tuple form supported by
    # SamplerV2.  When the shots argument is omitted from the tuple, the
    # default above is used.
    pub_with_shots = [
        (compiled_circuit, None, shots),
    ]
    if apply_mitigation:
        pub_with_shots.extend(
            [
                (cal_0_t, None, cal_shots),
                (cal_1_t, None, cal_shots),
            ]
        )
    job = sampler.run(pub_with_shots)
    job_id = getattr(job, "job_id", lambda: None)()
    print(f"  job_id={job_id}  status={job.status()}", flush=True)

    # Block until completion.  For long queues the user can Ctrl-C and
    # resume by re-running (the job lives on the IBM side for 24h).
    result = job.result()

    main_pub = result[0]
    counts_raw = _extract_bit_counts(
        main_pub.data.meas if hasattr(main_pub.data, "meas") else main_pub.data,
        aux_qubit_index,
    )

    calib_payload: dict[str, Any] = {"applied": False}
    counts_out = counts_raw
    if apply_mitigation:
        cal_0_pub = result[1]
        cal_1_pub = result[2]
        c0 = _extract_bit_counts(
            cal_0_pub.data.meas if hasattr(cal_0_pub.data, "meas") else cal_0_pub.data,
            0,
        )
        c1 = _extract_bit_counts(
            cal_1_pub.data.meas if hasattr(cal_1_pub.data, "meas") else cal_1_pub.data,
            0,
        )
        cal_total_0 = c0.get("0", 0) + c0.get("1", 0)
        cal_total_1 = c1.get("0", 0) + c1.get("1", 0)
        if cal_total_0 > 0 and cal_total_1 > 0:
            e10 = c0.get("1", 0) / cal_total_0
            e01 = c1.get("0", 0) / cal_total_1
            confusion = np.array(
                [[1.0 - e10, e01], [e10, 1.0 - e01]], dtype=np.float64
            )
            counts_out = _mitigate_single_qubit(counts_raw, confusion)
            calib_payload = {
                "applied": True,
                "cal_shots": cal_shots,
                "cal_physical_qubit": int(aux_phys),
                "c0_counts": c0,
                "c1_counts": c1,
                "e10_meas1_given_state0": float(e10),
                "e01_meas0_given_state1": float(e01),
                "confusion_matrix": confusion.tolist(),
            }
        else:
            calib_payload = {"applied": False, "reason": "empty calibration"}

    return counts_out, counts_raw, calib_payload, job_id


def _resolve_aux_physical_qubit(compiled_circuit, aux_qubit_index: int, report=None) -> int:
    """Return the IBM physical qubit ID that hosts the ``aux_qubit_index`` of
    the pre-transpilation circuit.

    The Heron pass manager is invoked with ``initial_layout=selected`` where
    ``selected`` is the calibration-aware subset, in the same order as the
    virtual qubits.  The transpiler may re-route some qubits due to heavy-hex
    topology, but for the QSVT projector the AUX is never swapped into a
    different slot, so the initial-layout assignment is a reliable proxy.

    We first try the transpiled circuit's own ``layout`` (using
    ``get_virtual_bits()`` for a Qubit-object-aware lookup), then fall back
    to ``report.selected_physical_qubits[aux_qubit_index]``.
    """

    # Try the transpiled circuit's layout first
    try:
        layout = getattr(compiled_circuit, "layout", None)
        if layout is not None:
            initial = getattr(layout, "initial_layout", None)
            if initial is not None:
                # ``initial_layout[virt_qubit]`` -> layout position.  Need to
                # find the original aux_qubit Qubit object in the layout.
                # The Layout's ``get_virtual_bits()`` returns a dict
                # ``{virtual_Qubit: physical_layout_index}``.
                vb = initial.get_virtual_bits()
                # Try direct integer lookup first
                if aux_qubit_index in vb:
                    return int(vb[aux_qubit_index])
                # Else find the Qubit with matching .index
                for vq, phys in vb.items():
                    if getattr(vq, "index", None) == aux_qubit_index:
                        return int(phys)
    except Exception:
        pass

    # Fall back to the report's selected_physical_qubits list (the IBM qubit
    # IDs from the calibration-aware selector, in virtual-qubit order).
    if report is not None:
        try:
            selected = list(getattr(report, "selected_physical_qubits", []) or [])
            if aux_qubit_index < len(selected):
                return int(selected[aux_qubit_index])
        except Exception:
            pass
    return int(aux_qubit_index)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run_boston_qpu(
    backend_name: str,
    degree: int,
    shots: int,
    target_loss_fraction: float,
    n_classical_scenarios: int,
    classical_seed: int,
    output: str,
    api_key_env: str,
    channel: str | None,
    optimization_level: int,
    seed_transpiler: int,
    apply_mitigation: bool,
    cal_shots: int,
    token_file: str | None,
    use_pyzx: bool,
) -> dict:
    t0 = time.time()
    circuit, meta = build_finnish_mortgage_qsvt_circuit(
        degree=degree,
        target_loss_fraction=target_loss_fraction,
    )

    # Measure only the AUX qubit (the QSVT answer bit).
    aux_idx = int(meta["aux_qubit_index"])
    measured = add_measurements(circuit, qubits=[aux_idx])

    backend = load_ibm_backend(
        backend_name=backend_name,
        api_key_env=api_key_env,
        channel=channel,
        token_file=token_file,
    )
    config = HeronCompileConfig(
        backend_name=backend_name,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        calibration_aware_layout=True,
        api_key_env=api_key_env,
        channel=channel,
        use_pyzx=use_pyzx,
    )
    compiled, report = compile_for_backend(measured, backend, config=config)

    counts_mit, counts_raw, calib, job_id = submit_main_and_mitigate(
        compiled_circuit=compiled,
        backend=backend,
        shots=shots,
        aux_qubit_index=aux_idx,
        apply_mitigation=apply_mitigation,
        cal_shots=cal_shots,
        report=report,
    )

    total = sum(v for k, v in counts_mit.items() if k in ("0", "1")) or 1
    prob_one_mit = counts_mit.get("1", 0) / total
    total_raw = sum(v for k, v in counts_raw.items() if k in ("0", "1")) or 1
    prob_one_raw = counts_raw.get("1", 0) / total_raw

    classical = classical_reference(
        meta=meta,
        n_scenarios=n_classical_scenarios,
        seed=classical_seed,
    )
    classical_tail = float(classical["tail_at_target_loss"])

    def _err(p, c):
        abs_e = abs(p - c)
        rel_e = abs_e / abs(c) if abs(c) > 1e-12 else float("nan")
        return abs_e, rel_e

    abs_raw, rel_raw = _err(prob_one_raw, classical_tail)
    abs_mit, rel_mit = _err(prob_one_mit, classical_tail)

    p = prob_one_mit
    se = float(np.sqrt(p * (1.0 - p) / total)) if total else 0.0
    ci_95 = (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))

    payload = {
        "backend_name": backend_name,
        "degree": degree,
        "shots": shots,
        "apply_mitigation": apply_mitigation,
        "cal_shots": cal_shots if apply_mitigation else None,
        "job_id": job_id,
        "dataset": meta,
        "precompile": {
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "count_ops": {str(k): int(v) for k, v in circuit.count_ops().items()},
        },
        "compile_report": report.as_dict(),
        "qpu_counts_raw": counts_raw,
        "qpu_counts_mitigated": counts_mit,
        "readout_calibration": calib,
        "qpu_total_shots": total,
        "qpu_p_loss_gt_target_raw": float(prob_one_raw),
        "qpu_p_loss_gt_target": float(prob_one_mit),
        "qpu_ci_95_p_loss_gt": [float(ci_95[0]), float(ci_95[1])],
        "classical": classical,
        "benchmark": {
            "metric": "P(loss > target_loss)",
            "classical_value": classical_tail,
            "quantum_value_raw": float(prob_one_raw),
            "quantum_value": float(prob_one_mit),
            "abs_error_raw": float(abs_raw),
            "rel_error_raw": float(rel_raw),
            "abs_error": float(abs_mit),
            "rel_error": float(rel_mit),
        },
        "runtime_seconds": time.time() - t0,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_token_file() -> str:
    return os.environ.get("IBM_API_KEY_FILE", DEFAULT_TOKEN_FILE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, submit, and benchmark the Finnish mortgage QSVT circuit "
            "on the real IBM QPU, scored against classical VaR/CVaR ground truth. "
            "Uses real Chebyshev-QSP phases (qsvt.approximator) and reads the "
            "AUX qubit (the QSVT projector answer bit)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend-name", default="ibm_boston")
    parser.add_argument(
        "--degree", type=int, default=8,
        help="QSVT polynomial degree. 8 is the sweet spot before noise dominates on Heron.",
    )
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--target-loss-fraction", type=float, default=0.5)
    parser.add_argument(
        "--n-classical-scenarios",
        type=int,
        default=200_000,
        help="Number of Monte Carlo scenarios for the classical reference.",
    )
    parser.add_argument("--classical-seed", type=int, default=0)
    parser.add_argument("--output", default="results/boston_qpu_k17.json")
    parser.add_argument("--api-key-env", default="IBM_API_KEY")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=42)
    parser.add_argument(
        "--use-pyzx",
        action="store_true",
        help=(
            "Run an extra pyzx.basic_optimization pass on the routed circuit. "
            "Helps on small/medium circuits but adds routing overhead on the "
            "K=17 QSVT (depth increases on Heron's heavy-hex).  Default: off."
        ),
    )
    parser.add_argument(
        "--no-mitigation",
        action="store_true",
        help="Skip the per-qubit readout calibration and mitigation round.",
    )
    parser.add_argument(
        "--cal-shots",
        type=int,
        default=512,
        help="Shots per calibration prep circuit (|0> and |1>).",
    )
    parser.add_argument(
        "--token-file",
        default=_default_token_file(),
        help=(
            "Path to a local file containing the IBM Quantum token on its "
            "first non-comment line.  Pass an empty string to disable the "
            "file fallback and only use $IBM_API_KEY."
        ),
    )
    args = parser.parse_args(argv)

    token_file = args.token_file if args.token_file else None
    payload = run_boston_qpu(
        backend_name=args.backend_name,
        degree=args.degree,
        shots=args.shots,
        target_loss_fraction=args.target_loss_fraction,
        n_classical_scenarios=args.n_classical_scenarios,
        classical_seed=args.classical_seed,
        output=args.output,
        api_key_env=args.api_key_env,
        channel=args.channel,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
        apply_mitigation=not args.no_mitigation,
        cal_shots=args.cal_shots,
        token_file=token_file,
        use_pyzx=args.use_pyzx,
    )

    summary = {
        "output": args.output,
        "backend_name": payload["backend_name"],
        "job_id": payload["job_id"],
        "compiled_depth": payload["compile_report"]["depth"],
        "compiled_two_qubit_proxy": payload["compile_report"]["two_qubit_depth_proxy"],
        "qpu_p_loss_gt_target_raw": payload["qpu_p_loss_gt_target_raw"],
        "qpu_p_loss_gt_target_mitigated": payload["qpu_p_loss_gt_target"],
        "classical_p_loss_gt_target": payload["benchmark"]["classical_value"],
        "classical_var_95": payload["classical"].get("var_0_95"),
        "classical_cvar_95": payload["classical"].get("cvar_0_95"),
        "abs_error_raw": payload["benchmark"]["abs_error_raw"],
        "rel_error_raw": payload["benchmark"]["rel_error_raw"],
        "abs_error_mitigated": payload["benchmark"]["abs_error"],
        "rel_error_mitigated": payload["benchmark"]["rel_error"],
        "readout_e10": (
            payload["readout_calibration"].get("e10_meas1_given_state0")
            if payload["readout_calibration"].get("applied")
            else None
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
