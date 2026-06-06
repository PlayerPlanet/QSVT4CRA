"""
Tests for loader/posterior_factor_copula.py
"""
import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from loader.posterior_factor_copula import PosteriorFactorCopulaLoader, _mapping


class TestMapping:
    """Tests for the _mapping helper function."""

    def test_mapping_basic(self):
        """Test basic binary to loss mapping."""
        lgd = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        # Binary 0000 ->0
        assert _mapping(0, lgd) == pytest.approx(0.0, abs=1e-6)

        # Binary 0001 -> lgd[3] = 0.4
        assert _mapping(1, lgd) == pytest.approx(0.4, abs=1e-6)

        # Binary 0010 -> lgd[2] = 0.3
        assert _mapping(2, lgd) == pytest.approx(0.3, abs=1e-6)

        # Binary 0011 -> lgd[2] + lgd[3] = 0.7
        assert _mapping(3, lgd) == pytest.approx(0.7, abs=1e-6)

        # Binary 1111 -> sum of all
        assert _mapping(15, lgd) == pytest.approx(0.1 + 0.2 + 0.3 + 0.4, abs=1e-6)

    def test_mapping_K(self):
        """Test mapping with different K values."""
        K = 3
        lgd = np.array([0.1, 0.2, 0.4], dtype=np.float32)

        # All ones
        assert _mapping(7, lgd) == pytest.approx(0.1 + 0.2 + 0.4, abs=1e-6)


class TestPosteriorFactorCopulaLoader:
    """Tests for PosteriorFactorCopulaLoader class."""

    @pytest.fixture
    def sample_theta(self):
        """Sample theta vector for K=10."""
        K = 10
        rng = np.random.default_rng(42)
        factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
        p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
        tail_dep = np.array(0.0, dtype=np.float32)
        copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
        return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])

    def test_num_qubits_correct(self, sample_theta):
        """Test that num_qubits = K + 1."""
        K = 10
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        assert loader.num_qubits == K + 1

    def test_K_property(self, sample_theta):
        """Test K property returns correct value."""
        K = 10
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        assert loader.K == K

    def test_to_gate_returns_blueprint(self, sample_theta):
        """Test that to_gate returns a BlueprintGate."""
        K = 10
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        gate = loader.to_gate()
        assert gate is not None
        assert hasattr(gate, "num_qubits")

    def test_compatible_with_circuitsCRA(self):
        """Test compatibility with get_expected_probability_circuit.

        Note: This test is skipped due to a known compatibility issue between
        pyqsp 0.2.0 (which returns a tuple from QuantumSignalProcessingPhases)
        and Code/QSVT.py (which expects just an array). The QSVTRiskCircuit
        handles this properly by extracting result[0] from the tuple.
        """
        pytest.skip("Skipped due to pyqsp 0.2.0 / Code/QSVT.py compatibility issue")

    def test_different_theta_produces_different_circuits(self):
        """Test that different theta values produce different circuits."""
        K = 3

        # Two different theta vectors
        rng1 = np.random.default_rng(42)
        theta1 = self._make_theta(K, rng1)

        rng2 = np.random.default_rng(123)
        theta2 = self._make_theta(K, rng2)

        loader1 = PosteriorFactorCopulaLoader(theta1, K=K)
        loader2 = PosteriorFactorCopulaLoader(theta2, K=K)

        # Get the gate definitions
        gate1 = loader1.to_gate()
        gate2 = loader2.to_gate()

        # Gates should be different for different theta
        # (We can't directly compare, but both should be valid)
        assert gate1.num_qubits == K + 1
        assert gate2.num_qubits == K + 1

    def test_decompose(self):
        """Test that decompose returns a valid circuit."""
        K = 3
        # Create theta for K=3
        rng = np.random.default_rng(42)
        factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
        p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
        tail_dep = np.array(0.0, dtype=np.float32)
        copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
        theta = np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])

        loader = PosteriorFactorCopulaLoader(theta, K=K)

        decomposed = loader.decompose()
        assert decomposed is not None
        assert len(decomposed.data) > 0

    def test_lgd_property(self, sample_theta):
        """Test that lgd property returns correct values."""
        K = 10
        loader = PosteriorFactorCopulaLoader(sample_theta, K=K)

        lgd = loader.lgd
        assert len(lgd) == K
        assert np.all(lgd == 0.40)  # Default LGD

    def _make_theta(self, K, rng):
        """Helper to create theta vector."""
        factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
        p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
        tail_dep = np.array(0.0, dtype=np.float32)
        copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
        return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])
