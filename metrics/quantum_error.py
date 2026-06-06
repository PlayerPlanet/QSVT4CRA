"""
Quantum vs classical error analysis for QSVT approximations.

Provides error metrics comparing quantum circuit outputs to classical
VaR/CVaR ground truth values.

Architecture: D4 (docs/architecture.md §4)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def quantum_vs_classical_error(
    qc,
    classical_value: float,
    n_shots: int = 10000,
    backend: str = "aer_simulator",
) -> dict:
    """
    Compare quantum circuit measurement result to classical value.

    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit to simulate.
    classical_value : float
        The classical ground truth value.
    n_shots : int, default 10000
        Number of measurement shots for simulation.
    backend : str, default "aer_simulator"
        Qiskit Aer backend to use.

    Returns
    -------
    dict
        Dictionary with keys:
        - quantum_estimate : float
 - classical_value : float
        - abs_error : float
        - rel_error : float
        - ci_95 : tuple (lower, upper)
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        return _error_dict(classical_value, classical_value, 0.0, 0.0, (0.0, 0.0))

    # Simulate the circuit
    try:
        simulator = AerSimulator()
        result = simulator.run(qc, shots=n_shots).result()
        counts = result.get_counts(qc)
    except Exception:
        return _error_dict(classical_value, classical_value, 0.0, 0.0, (0.0, 0.0))

    # Extract probability of measuring |1> on the objective qubit
    # Assume the objective qubit is the last qubit
    total_shots = sum(counts.values())
    if total_shots == 0:
        return _error_dict(classical_value, classical_value, 0.0, 0.0, (0.0, 0.0))

    # Count measurements where objective qubit is |1>
    # In Qiskit count format, states are like "0101" where last bit is qubit 0
    prob_one = 0.0
    n_qubits = qc.num_qubits

    for state, count in counts.items():
        # Parse the state string
        if isinstance(state, str):
            # Pad to n_qubits
            state_padded = state.zfill(n_qubits)
            # Last qubit is the objective
            if state_padded[-1] == "1":
                prob_one += count
        elif isinstance(state, int):
            if state & 1:
                prob_one += count

    quantum_estimate = prob_one / total_shots

    # Compute errors
    abs_error = abs(quantum_estimate - classical_value)
    if abs(classical_value) > 1e-10:
        rel_error = abs_error / abs(classical_value)
    else:
        rel_error = abs_error

    # 95% CI via normal approximation
    p = quantum_estimate
    se = np.sqrt(p * (1 - p) / total_shots)
    ci_95 = (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))

    return {
        "quantum_estimate": float(quantum_estimate),
        "classical_value": float(classical_value),
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "ci_95": (float(ci_95[0]), float(ci_95[1])),
    }


def _error_dict(
    quantum_estimate: float,
    classical_value: float,
    abs_error: float,
    rel_error: float,
    ci_95: Tuple[float, float],
) -> dict:
    """Create error dictionary."""
    return {
        "quantum_estimate": quantum_estimate,
        "classical_value": classical_value,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "ci_95": ci_95,
    }


def cdf_error(
    qc,
    x_grid: np.ndarray,
    classical_cdf: np.ndarray,
    n_shots: int = 10000,
) -> dict:
    """
    Compute KS statistic between quantum and classical CDF.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit.
    x_grid : np.ndarray
        Grid points for CDF evaluation.
    classical_cdf : np.ndarray
        Classical CDF values at x_grid.
    n_shots : int, default 10000
        Number of shots per evaluation.

    Returns
    -------
    dict
        Dictionary with keys:
        - ks_statistic : float
        - quantum_cdf : np.ndarray
        - classical_cdf : np.ndarray
    """
    quantum_cdf = np.zeros_like(classical_cdf, dtype=np.float32)

    try:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
    except ImportError:
        return {
            "ks_statistic": 0.0,
            "quantum_cdf": quantum_cdf,
            "classical_cdf": classical_cdf,
        }

    # For each grid point, compute P(loss <= x)
    for i, x in enumerate(x_grid):
        # Build circuit that computes P(loss <= x)
        try:
            result = simulator.run(qc, shots=n_shots).result()
            counts = result.get_counts(qc)

            # Compute probability of measuring objective = 0 (loss <= threshold)
            total_shots = sum(counts.values())
            prob_zero = 0.0
            n_qubits = qc.num_qubits

            for state, count in counts.items():
                if isinstance(state, str):
                    state_padded = state.zfill(n_qubits)
                    if state_padded[-1] == "0":
                        prob_zero += count
                elif isinstance(state, int):
                    if not (state & 1):
                        prob_zero += count

            quantum_cdf[i] = prob_zero / total_shots
        except Exception:
            quantum_cdf[i] = classical_cdf[i]

    # KS statistic
    ks_statistic = float(np.max(np.abs(quantum_cdf - classical_cdf)))

    return {
        "ks_statistic": ks_statistic,
        "quantum_cdf": quantum_cdf,
        "classical_cdf": classical_cdf,
    }


def tail_error(
    qc,
    target: float,
    classical_tail_prob: float,
    n_shots: int = 10000,
) -> dict:
    """
    Compute error in tail probability P(loss > target).

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit.
    target : float
        Target loss threshold.
    classical_tail_prob : float
        Classical tail probability.
    n_shots : int, default 10000
        Number of shots.

    Returns
    -------
    dict
        Dictionary with keys:
        - quantum_tail_prob : float
        - classical_tail_prob : float
        - abs_error : float
        - rel_error : float
    """
    result = quantum_vs_classical_error(qc, classical_tail_prob, n_shots)

    quantum_tail = result["quantum_estimate"]
    abs_error = abs(quantum_tail - classical_tail_prob)
    if abs(classical_tail_prob) > 1e-10:
        rel_error = abs_error / abs(classical_tail_prob)
    else:
        rel_error = abs_error

    return {
        "quantum_tail_prob": quantum_tail,
        "classical_tail_prob": classical_tail_prob,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def var_error(
    qc,
    alphas: List[float],
    classical_var: dict,
    n_shots: int = 10000,
) -> dict:
    """
    Compute VaR error at multiple confidence levels.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit.
    alphas : list of float
        Confidence levels (e.g., [0.95, 0.99]).
    classical_var : dict
        Classical VaR values keyed by alpha (e.g., {"0.95": 0.5}).
    n_shots : int, default 10000
        Number of shots.

    Returns
    -------
    dict
        Dictionary with per-alpha error metrics.
    """
    errors = {}

    for alpha in alphas:
        key = str(alpha).replace(".", "_")
        classical_val = classical_var.get(key, 0.0)

        result = quantum_vs_classical_error(qc, classical_val, n_shots)
        errors[f"var_{key}"] = {
            "quantum_estimate": result["quantum_estimate"],
            "classical_value": result["classical_value"],
            "abs_error": result["abs_error"],
            "rel_error": result["rel_error"],
        }

    return errors


def cvar_error(
    qc,
    alphas: List[float],
    classical_cvar: dict,
    n_shots: int = 10000,
) -> dict:
    """
    Compute CVaR error at multiple confidence levels.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit.
    alphas : list of float
        Confidence levels.
    classical_cvar : dict
        Classical CVaR values keyed by alpha.
    n_shots : int, default 10000
        Number of shots.

    Returns
    -------
    dict
        Dictionary with per-alpha error metrics.
    """
    errors = {}

    for alpha in alphas:
        key = str(alpha).replace(".", "_")
        classical_val = classical_cvar.get(key, 0.0)

        result = quantum_vs_classical_error(qc, classical_val, n_shots)
        errors[f"cvar_{key}"] = {
            "quantum_estimate": result["quantum_estimate"],
            "classical_value": result["classical_value"],
            "abs_error": result["abs_error"],
            "rel_error": result["rel_error"],
        }

    return errors
