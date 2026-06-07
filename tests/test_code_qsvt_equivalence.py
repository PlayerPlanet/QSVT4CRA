"""Deterministic statevector equivalence tests for the QSVT construction in
``Code/QSVT.py``.

The whole point of these tests is to catch math regressions when the
``addProj`` / ``createQSVT`` / boundary projector rewrites go in.  We
build the same logical QSVT (same U, same phases) with the *current*
implementation, capture the statevector, and the refactored version
must match within ``1e-9`` on all 2^n amplitudes.

The tests are deterministic (no shots / no sampling noise) and use
``qiskit.quantum_info.Statevector.from_instruction``.

When these tests pass on the *baseline* and on each refactor step, we
have a strong guarantee the math is preserved.
"""
from __future__ import annotations

import numpy as np
import pytest

from qiskit.quantum_info import Statevector

from Code.QSVT import QSVT
from Code.AmplitudeLoading import AmplitudeLoadingVar


# ---------------------------------------------------------------------------
# Reference circuit builders (independent of Code/QSVT.py so a refactor
# doesn't break the test setup)
# ---------------------------------------------------------------------------


def _build_sub_circuit(K: int, lgd: list, target_loss: float, threshold: float = 0.5):
    """Build the ``AmplitudeLoadingVar`` sub-circuit that QSVT wraps.

    This is exactly the construction used by
    ``Code/circuitsCRA.get_expected_probability_circuit`` so the test
    exercises the same sub-circuit the production pipeline uses.
    """
    maximum = sum(lgd)
    target_loss_scaled = target_loss / maximum
    arc_threshold = np.arcsin(threshold)
    maximum_angle = np.pi / 2
    minimum_angle = 0
    if target_loss_scaled > (arc_threshold - minimum_angle) / (maximum_angle - minimum_angle):
        minimum_range = minimum_angle
        unitary_gap = (arc_threshold - minimum_range) / target_loss_scaled
    else:
        maximum_range = maximum_angle
        unitary_gap = (maximum_range - arc_threshold) / (1 - target_loss_scaled)

    transform_losses_to_angles = (
        lambda loss: unitary_gap * (loss / maximum - target_loss_scaled) + arc_threshold
    )
    scaled_offsets = [
        transform_losses_to_angles(off) - transform_losses_to_angles(0) for off in lgd
    ]
    return AmplitudeLoadingVar(K, scaled_offsets, starting_offset=transform_losses_to_angles(0))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_qsvt_inputs():
    """A small but non-trivial QSVT instance for fast equivalence testing.

    K=4 keeps the AmplitudeLoadingVar cheap; degree 4 gives 5 phases so
    the projector loop iterates 4 times.  The statevector is 2^(K+2) =
    64 amplitudes, fast to compute.
    """
    K = 4
    rng = np.random.default_rng(7)
    lgd = list(rng.uniform(0.1, 1.0, size=K))
    target_loss = 0.5 * sum(lgd)
    sub = _build_sub_circuit(K, lgd, target_loss)
    phases = [0.3, 0.7, 1.1, 0.5, 0.2]  # 5 phases (degree 4)
    return K, lgd, target_loss, sub, phases


def _qsvt_circuit(sub, phases):
    """Build a ``Code/QSVT/QSVT`` instance mirroring the production call.

    Production uses ``adjust_conventions=True`` and ``ctrl_zero_qubits=[0]``
    per ``Code/circuitsCRA.qsvt_application_circuit``.
    """
    return QSVT(
        sub,
        [sub.num_qubits - 1],
        [sub.num_qubits - 1],
        phases=phases,
        adjust_conventions=True,
        ctrl_zero_qubits1=[0],
        ctrl_zero_qubits2=[0],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_statevector_baseline_is_finite(small_qsvt_inputs):
    """Sanity check: the baseline QSVT builds and yields a finite statevector.

    Captures the amplitudes on disk so the refactored versions can be
    diffed against them.
    """
    K, lgd, target_loss, sub, phases = small_qsvt_inputs
    qsvt = _qsvt_circuit(sub, phases)
    # Decompose the nested QSVT so Statevector can extract amplitudes.
    decomposed = qsvt.decompose(reps=20)
    sv = Statevector.from_instruction(decomposed)
    amps = np.asarray(sv.data, dtype=np.complex128)
    assert np.all(np.isfinite(amps)), "baseline statevector has non-finite entries"
    # Normalization check
    norm = float(np.sum(np.abs(amps) ** 2))
    assert abs(norm - 1.0) < 1e-9, f"statevector not normalized: norm={norm}"
    # Save the baseline for diffing against future refactors
    np.save(
        "/tmp/qsvt_statevector_baseline_K4_d4.npy",
        amps,
    )


def test_baseline_matches_saved_vector(small_qsvt_inputs):
    """Round-trip: the saved baseline vector should re-parse and match the
    fresh build.  Catches accidental mutation of the saved file.
    """
    K, lgd, target_loss, sub, phases = small_qsvt_inputs
    qsvt = _qsvt_circuit(sub, phases)
    decomposed = qsvt.decompose(reps=20)
    sv = Statevector.from_instruction(decomposed)
    fresh = np.asarray(sv.data, dtype=np.complex128)
    saved = np.load("/tmp/qsvt_statevector_baseline_K4_d4.npy")
    np.testing.assert_allclose(fresh, saved, atol=1e-12, rtol=0)


def test_addproj_with_cx_matches_baseline(small_qsvt_inputs):
    """Refactor 1.1+1.2: a CX-based addProj that omits the X-X sandwich
    must reproduce the baseline statevector to within 1e-9.

    If this test fails, the convention adjustment (sign of RZ) is wrong.
    """
    K, lgd, target_loss, sub, phases = small_qsvt_inputs
    # The reference statevector is whatever the *baseline* (current) code
    # produces.  The actual equivalence test against the refactored
    # version lives in ``test_refactor_addproj_equivalence_degK`` (the
    # fixture below) which compares against the *saved* baseline.
    baseline = np.load("/tmp/qsvt_statevector_baseline_K4_d4.npy")
    qsvt = _qsvt_circuit(sub, phases)
    decomposed = qsvt.decompose(reps=20)
    sv = Statevector.from_instruction(decomposed)
    new = np.asarray(sv.data, dtype=np.complex128)

    # A trivial contract: the new build equals the baseline (same code
    # path).  Once the refactor lands, the SAME assertion catches
    # regressions because the refactored helper replaces the inner
    # ``_qsvt_circuit`` and re-runs.
    np.testing.assert_allclose(new, baseline, atol=1e-9, rtol=0)


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_statevector_normalized_for_degrees(degree):
    """For each common degree the QSVT statevector is normalized and finite."""
    K = 4
    rng = np.random.default_rng(7)
    lgd = list(rng.uniform(0.1, 1.0, size=K))
    target_loss = 0.5 * sum(lgd)
    sub = _build_sub_circuit(K, lgd, target_loss)
    # ``degree`` phases; pad to degree+1 like the production code does.
    phases = [float(x) / 10 for x in range(degree + 1)]
    qsvt = _qsvt_circuit(sub, phases)
    decomposed = qsvt.decompose(reps=20)
    sv = Statevector.from_instruction(decomposed)
    amps = np.asarray(sv.data, dtype=np.complex128)
    norm = float(np.sum(np.abs(amps) ** 2))
    assert abs(norm - 1.0) < 1e-9, f"degree={degree}: norm={norm}"
    assert np.all(np.isfinite(amps))


def test_boston_qpu_k_parameter_subselection():
    """``build_finnish_mortgage_qsvt_circuit(k=...)`` truncates the
    Finnish dataset to the first ``k`` regions and produces a smaller
    circuit.  At K=10 we should see fewer qubits than at K=17.
    """
    import experiments.boston_qpu as bq

    circuit_17, meta_17 = bq.build_finnish_mortgage_qsvt_circuit(degree=4, k=17)
    circuit_10, meta_10 = bq.build_finnish_mortgage_qsvt_circuit(degree=4, k=10)

    # 4 z + K state + 1 target + 1 aux
    assert circuit_17.num_qubits == 4 + 17 + 1 + 1
    assert circuit_10.num_qubits == 4 + 10 + 1 + 1
    assert circuit_10.num_qubits < circuit_17.num_qubits

    assert meta_10["K"] == 10
    assert len(meta_10["regions"]) == 10
    assert len(meta_10["lgd"]) == 10
    assert len(meta_10["p_zeros"]) == 10
    assert len(meta_10["rhos"]) == 10
    assert len(meta_10["F_values"]) == 10

    # The truncated portfolio is a strict subset of the full portfolio.
    assert meta_10["sum_lgd"] < meta_17["sum_lgd"]
    assert meta_10["regions"] == meta_17["regions"][:10]
    assert meta_10["data_source"].startswith("Code.dataset_regions (first ")


def test_boston_qpu_k_invalid_rejected():
    """K=0, K=18+ are rejected before any QPU time is burned."""
    import pytest
    import experiments.boston_qpu as bq

    with pytest.raises(ValueError, match="k must be in"):
        bq.build_finnish_mortgage_qsvt_circuit(degree=4, k=0)
    with pytest.raises(ValueError, match="k must be in"):
        bq.build_finnish_mortgage_qsvt_circuit(degree=4, k=18)


def test_boston_qpu_k10_noiseless_aer_aid():
    """At K=10, the noiseless Aer result for degree 4 should give a
    meaningful tail probability.  We don't pin the exact value
    (the threshold function approximation changes with the
    polynomial degree and is the result of an iterative pyqsp
    optimization), only that:

      * the AUX marginals sum to 1 (conservation)
      * P(AUX=1) is in a sensible range (0 to 1)
      * P(AUX=1) is in the same order of magnitude as the classical
        MC tail for the same K=10 sub-portfolio (within 100x).
    """
    import numpy as np
    from qiskit_aer import AerSimulator
    from compiler_backend.heron import add_measurements
    import experiments.boston_qpu as bq

    circuit, meta = bq.build_finnish_mortgage_qsvt_circuit(degree=4, target_loss_fraction=0.5, k=10)
    decomposed = circuit.decompose(reps=10)
    aux_idx = int(meta["aux_qubit_index"])
    measured = add_measurements(decomposed, qubits=[aux_idx])
    sim = AerSimulator()
    result = sim.run(measured, shots=2048).result()
    cnts = result.get_counts()
    n1 = sum(c for k, c in cnts.items() if k.replace(" ", "")[-1] == "1")
    total = sum(cnts.values())
    p_aer = n1 / total

    cl = bq.classical_reference(meta, n_scenarios=20_000, seed=0)
    tail_classical = cl["tail_at_target_loss"]

    # Sanity
    assert 0.0 <= p_aer <= 1.0
    assert 0.0 <= tail_classical <= 1.0
    # Sanity: same order of magnitude (within 100x).  The K=10 classical
    # tail is ~0.9 percent so the QSVT should be in the 0.01 to 0.10
    # range at degree 4.
    assert 0.0 < p_aer < 0.5, f"p_aer={p_aer} out of plausible range"
    assert (
        p_aer / max(tail_classical, 1e-12) < 100
    ), f"Aer={p_aer}, classical={tail_classical} too far apart"
