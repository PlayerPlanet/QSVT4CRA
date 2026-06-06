"""
Tests for qsvt/circuit.py
"""
import numpy as np
import pytest

from loader.posterior_factor_copula import PosteriorFactorCopulaLoader
from qsvt.circuit import QSVTRiskCircuit


class TestQSVTRiskCircuit:
    """Tests for QSVTRiskCircuit class."""

    @pytest.fixture
    def sample_theta(self):
        """Sample theta vector for K=3."""
        K = 3
        rng = np.random.default_rng(42)
        factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
        p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
        tail_dep = np.array(0.0, dtype=np.float32)
        copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
        return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])

    def test_full_circuit_construction(self, sample_theta):
        """Test that full circuit can be constructed."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        circuit = QSVTRiskCircuit(
            loader=loader,
            target_loss=0.5,
            degree=8,
            threshold=0.5,
        )

        assert circuit is not None
        assert circuit.num_qubits > 0

    def test_num_qubits(self, sample_theta):
        """Test circuit num_qubits property."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        circuit = QSVTRiskCircuit(
            loader=loader,
            target_loss=0.5,
            degree=8,
        )

        assert circuit.num_qubits == K + 2  # K state + 1 target + 1 aux

    def test_K_property(self, sample_theta):
        """Test K property."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        circuit = QSVTRiskCircuit(
            loader=loader,
            target_loss=0.5,
            degree=8,
        )

        assert circuit.K == K

    def test_to_gate(self, sample_theta):
        """Test to_gate method."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        circuit = QSVTRiskCircuit(
            loader=loader,
            target_loss=0.5,
            degree=8,
        )

        gate = circuit.to_gate()
        assert gate is not None
        assert hasattr(gate, "num_qubits")

    def test_decompose(self, sample_theta):
        """Test decompose method."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        circuit = QSVTRiskCircuit(
            loader=loader,
            target_loss=0.5,
            degree=8,
        )

        decomposed = circuit.decompose()
        assert decomposed is not None

    def test_different_degrees(self, sample_theta):
        """Test with different polynomial degrees."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        for degree in [4, 8, 16]:
            circuit = QSVTRiskCircuit(
                loader=loader,
                target_loss=0.5,
                degree=degree,
            )
            assert circuit.num_qubits > 0

    def test_different_target_losses(self, sample_theta):
        """Test with different target loss values."""
        K = 3
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        for target_loss in [0.3, 0.5, 0.7]:
            circuit = QSVTRiskCircuit(
                loader=loader,
                target_loss=target_loss,
                degree=8,
            )
            assert circuit.num_qubits > 0
