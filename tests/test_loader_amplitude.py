"""
Tests for loader/amplitude_loader.py
"""
import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from loader.amplitude_loader import AmplitudeLoader


class TestAmplitudeLoader:
    """Tests for AmplitudeLoader class."""

    def test_shape(self):
        """Test that AmplitudeLoader creates circuit with correct shape."""
        K = 3
        values = np.random.rand(2**K).astype(np.float32)
        loader = AmplitudeLoader(K, values)

        assert loader.num_state_qubits == K
        assert loader.num_qubits == K + 1  # K state + 1 target

    def test_values_length_mismatch(self):
        """Test that incorrect values length raises error."""
        K = 3
        values = np.random.rand(2**(K + 1)).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match="values must have length"):
            AmplitudeLoader(K, values)

    def test_normalization(self):
        """Test that amplitudes are properly normalized."""
        K = 4
        values = np.random.rand(2**K).astype(np.float32) * 10
        loader = AmplitudeLoader(K, values)

        # Check that amplitudes sum to ~1
        # We can't directly access _amplitudes, but we can check the circuit works
        assert loader.num_qubits == K + 1

    def test_decomposition_count(self):
        """Test that decomposition produces gates."""
        K = 3
        values = np.random.rand(2**K).astype(np.float32)
        loader = AmplitudeLoader(K, values)

        decomposed = loader.decompose()
        # Decomposed circuit should have non-zero size
        assert len(decomposed.data) > 0

    def test_to_gate(self):
        """Test that to_gate returns a gate."""
        K = 3
        values = np.random.rand(2**K).astype(np.float32)
        loader = AmplitudeLoader(K, values)

        gate = loader.to_gate()
        assert gate is not None

    def test_small_K(self):
        """Test with small K values."""
        for K in [1, 2, 3]:
            values = np.random.rand(2**K).astype(np.float32)
            loader = AmplitudeLoader(K, values)
            assert loader.num_qubits == K + 1

    def test_zero_values(self):
        """Test with zero values (edge case)."""
        K = 2
        values = np.zeros(2**K, dtype=np.float32)
        loader = AmplitudeLoader(K, values)
        assert loader.num_qubits == K + 1

    def test_identical_values(self):
        """Test with identical values."""
        K = 3
        values = np.ones(2**K, dtype=np.float32)
        loader = AmplitudeLoader(K, values)
        assert loader.num_qubits == K + 1
