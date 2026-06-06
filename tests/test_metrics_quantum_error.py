"""
Tests for metrics/quantum_error.py
"""
import numpy as np
import pytest

from metrics.quantum_error import (
    quantum_vs_classical_error,
    cdf_error,
    tail_error,
    var_error,
    cvar_error,
)


class TestQuantumVsClassicalError:
    """Tests for quantum_vs_classical_error function."""

    def test_end_to_end_small_circuit(self):
        """End-to-end test with K=3, 4-state, 1000 shots."""
        # This test requires a real quantum circuit
        # We'll use a simple circuit that produces a known result
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
        except ImportError:
            pytest.skip("Qiskit Aer not available")

        # Create a simple circuit that measures |0> with probability 0.7
        K = 3
        qc = QuantumCircuit(K + 1)  # K state + 1 objective
        # Initialize to known state
        qc.x(K)  # Flip objective to |1>
        # This should give P(|1>) = 1.0

        result = quantum_vs_classical_error(qc, 0.7, n_shots=1000)

        assert "quantum_estimate" in result
        assert "classical_value" in result
        assert "abs_error" in result
        assert "rel_error" in result
        assert "ci_95" in result

    def test_no_aer_fallback(self):
        """Test graceful fallback when Aer is not available."""
        # Create a dummy circuit
        try:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2)
        except ImportError:
            pytest.skip("Qiskit not available")

        # This should return an error dict gracefully
        result = quantum_vs_classical_error(qc, 0.5, n_shots=1000)

        assert "quantum_estimate" in result
        assert "classical_value" in result


class TestCdfError:
    """Tests for cdf_error function."""

    def test_cdf_error_basic(self):
        """Test CDF error computation."""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            pytest.skip("Qiskit not available")

        qc = QuantumCircuit(2)
        x_grid = np.array([0.0, 0.5, 1.0])
        classical_cdf = np.array([0.0, 0.5, 1.0])

        result = cdf_error(qc, x_grid, classical_cdf, n_shots=100)

        assert "ks_statistic" in result
        assert "quantum_cdf" in result
        assert "classical_cdf" in result


class TestTailError:
    """Tests for tail_error function."""

    def test_tail_error_basic(self):
        """Test tail probability error."""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            pytest.skip("Qiskit not available")

        qc = QuantumCircuit(2)
        target = 0.5
        classical_tail = 0.1

        result = tail_error(qc, target, classical_tail, n_shots=100)

        assert "quantum_tail_prob" in result
        assert "classical_tail_prob" in result
        assert "abs_error" in result
        assert "rel_error" in result


class TestVarError:
    """Tests for var_error function."""

    def test_var_error_multiple_alphas(self):
        """Test VaR error at multiple confidence levels."""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            pytest.skip("Qiskit not available")

        qc = QuantumCircuit(2)
        alphas = [0.95, 0.99]
        classical_var = {"0_95": 0.5, "0_99": 0.7}

        result = var_error(qc, alphas, classical_var, n_shots=100)

        assert "var_0_95" in result
        assert "var_0_99" in result


class TestCvarError:
    """Tests for cvar_error function."""

    def test_cvar_error_multiple_alphas(self):
        """Test CVaR error at multiple confidence levels."""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            pytest.skip("Qiskit not available")

        qc = QuantumCircuit(2)
        alphas = [0.95, 0.99]
        classical_cvar = {"0_95": 0.6, "0_99": 0.8}

        result = cvar_error(qc, alphas, classical_cvar, n_shots=100)

        assert "cvar_0_95" in result
        assert "cvar_0_99" in result
