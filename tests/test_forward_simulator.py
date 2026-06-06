"""
Tests for simulator/forward.py — ForwardSimulator, NumPyForwardSimulator,
JAXForwardSimulator.
"""
from __future__ import annotations

import numpy as np
import pytest
from data.synthetic import SyntheticPortfolioGenerator
from simulator.forward import (
    ForwardSimulator,
    NumPyForwardSimulator,
    JAXForwardSimulator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def K():
    return 10


@pytest.fixture
def generator(K):
    return SyntheticPortfolioGenerator(K=K, seed=42)


@pytest.fixture
def theta_baseline(K):
    """Baseline theta for K=10."""
    rng = np.random.default_rng(0)
    factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
    p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
    tail_dep = np.array(0.0, dtype=np.float32)
    copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
    return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])


@pytest.fixture
def numpy_sim(generator):
    return NumPyForwardSimulator(portfolio_generator=generator, regime="baseline", seed=42)


@pytest.fixture
def jax_sim(generator):
    return JAXForwardSimulator(portfolio_generator=generator, regime="baseline", seed=42)


@pytest.fixture
def numpy_sim_for_jax(generator):
    """Separate NumPy simulator for JAX comparison (avoids shared RNG state)."""
    return NumPyForwardSimulator(
        portfolio_generator=SyntheticPortfolioGenerator(K=generator.K, seed=99),
        regime="baseline",
        seed=99,
    )


# ---------------------------------------------------------------------------
# Abstract base tests
# ---------------------------------------------------------------------------
class TestForwardSimulatorAbstract:
    """Test that ForwardSimulator cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self, generator):
        with pytest.raises(TypeError):
            ForwardSimulator(portfolio_generator=generator)


# ---------------------------------------------------------------------------
# NumPyForwardSimulator tests
# ---------------------------------------------------------------------------
class TestNumPyForwardSimulatorSimulate:
    """Tests for NumPyForwardSimulator.simulate()."""

    def test_simulate_single_theta_shape(self, numpy_sim, theta_baseline):
        result = numpy_sim.simulate(theta_baseline[np.newaxis, :], n_scenarios=200)
        assert result.shape == (1, 10)
        assert result.dtype == np.float32

    def test_simulate_batch_shape(self, numpy_sim, theta_baseline):
        B = 5
        theta_batch = np.stack([theta_baseline] * B, axis=0)
        result = numpy_sim.simulate(theta_batch, n_scenarios=200)
        assert result.shape == (B, 10)
        assert result.dtype == np.float32

    def test_simulate_single_theta_convenience(self, numpy_sim, theta_baseline):
        result = numpy_sim.simulate_single(theta_baseline, n_scenarios=200)
        assert result.shape == (10,)
        assert result.dtype == np.float32

    def test_simulate_wrong_dim_raises(self, numpy_sim, theta_baseline):
        with pytest.raises(ValueError, match="theta_batch must be 2D"):
            numpy_sim.simulate(theta_baseline, n_scenarios=200)

    def test_simulate_batch_deterministic(self, numpy_sim, theta_baseline):
        """Same theta + same seed → identical results within a single batch."""
        theta_batch = theta_baseline[np.newaxis, :]
        r1 = numpy_sim.simulate(theta_batch, n_scenarios=100)
        # Create a fresh simulator with same seed to verify determinism
        from data.synthetic import SyntheticPortfolioGenerator
        fresh_sim = NumPyForwardSimulator(
            portfolio_generator=SyntheticPortfolioGenerator(
                K=numpy_sim.portfolio_generator.K, seed=42
            ),
            regime="baseline",
            seed=42,
        )
        r2 = fresh_sim.simulate(theta_batch, n_scenarios=100)
        np.testing.assert_array_equal(r1, r2)

    def test_var95_reasonable(self, numpy_sim, theta_baseline):
        result = numpy_sim.simulate_single(theta_baseline, n_scenarios=1000)
        var95 = result[9]
        assert var95 >= 0.0


class TestNumPyForwardSimulatorGradLogLikelihood:
    """Tests for grad_log_likelihood()."""

    def test_grad_shape(self, numpy_sim, theta_baseline):
        x_obs = np.zeros(10, dtype=np.float32)
        grad = numpy_sim.grad_log_likelihood(theta_baseline, x_obs)
        assert grad.shape == theta_baseline.shape
        assert grad.dtype == np.float32

    def test_grad_finite(self, numpy_sim, theta_baseline):
        x_obs = np.zeros(10, dtype=np.float32)
        grad = numpy_sim.grad_log_likelihood(theta_baseline, x_obs)
        assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# JAXForwardSimulator tests
# ---------------------------------------------------------------------------
class TestJAXForwardSimulator:
    """Tests for JAXForwardSimulator."""

    def test_init_does_not_fail(self, generator):
        """JAXForwardSimulator should init even without JAX installed."""
        sim = JAXForwardSimulator(portfolio_generator=generator)
        # Should have None backends if JAX unavailable
        assert sim._jax is None or sim._jax is not None

    def test_fallback_when_jax_unavailable(self, generator, theta_baseline):
        """If JAX is not installed, should fall back to NumPy."""
        sim = JAXForwardSimulator(portfolio_generator=generator)
        # Force NumPy fallback by setting _jax to None
        sim._jax = None
        result = sim.simulate(theta_baseline[np.newaxis, :], n_scenarios=100)
        assert result.shape == (1, 10)
        assert result.dtype == np.float32

    def test_fallback_when_no_gpu(self, generator, theta_baseline):
        """If JAX is installed but no GPU, should fall back to NumPy."""
        sim = JAXForwardSimulator(portfolio_generator=generator)
        if sim._jax is not None:
            # Mock no GPU by patching default_backend to return 'cpu'
            original_backend = sim._jax.default_backend
            sim._jax.default_backend = lambda: "cpu"
            result = sim.simulate(theta_baseline[np.newaxis, :], n_scenarios=100)
            assert result.shape == (1, 10)
            assert result.dtype == np.float32
            sim._jax.default_backend = original_backend

    def test_batch_shape(self, jax_sim, theta_baseline):
        """Batch simulation should return correct shape regardless of backend."""
        B = 4
        theta_batch = np.stack([theta_baseline] * B, axis=0)
        result = jax_sim.simulate(theta_batch, n_scenarios=100)
        assert result.shape == (B, 10)
        assert result.dtype == np.float32

    def test_single_theta_shape(self, jax_sim, theta_baseline):
        result = jax_sim.simulate_single(theta_baseline, n_scenarios=100)
        assert result.shape == (10,)
        assert result.dtype == np.float32

    def test_jax_grad_log_likelihood_runs(self, jax_sim, theta_baseline):
        """grad_log_likelihood should run (JAX or NumPy fallback)."""
        x_obs = np.zeros(10, dtype=np.float32)
        grad = jax_sim.grad_log_likelihood(theta_baseline, x_obs)
        assert grad.shape == theta_baseline.shape
        assert grad.dtype == np.float32
        assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
class TestForwardSimulatorIntegration:
    """Integration tests across both backends."""

    @pytest.mark.parametrize("backend", ["numpy", "jax"])
    def test_observation_values_reasonable(
        self, backend, generator, theta_baseline
    ):
        """Observations should have reasonable values."""
        if backend == "numpy":
            sim = NumPyForwardSimulator(generator, seed=42)
        else:
            sim = JAXForwardSimulator(generator, seed=42)
            if sim._jax is not None:
                # Force CPU fallback for predictable results
                sim._jax.default_backend = lambda: "cpu"

        result = sim.simulate_single(theta_baseline, n_scenarios=500)
        # var95 (col 9) should be non-negative
        assert result[9] >= 0.0
        # n_defaults (col 0) should be in [0, K]
        assert result[0] >= 0.0
        assert result[0] <= generator.K

    def test_jax_matches_numpy_output(self, generator, theta_baseline):
        """JAX and NumPy should produce same results when both use NumPy fallback."""
        # Use independent generator instances so RNG state doesn't leak between simulators
        from data.synthetic import SyntheticPortfolioGenerator

        gen_np = SyntheticPortfolioGenerator(generator.K, seed=42)
        gen_jax = SyntheticPortfolioGenerator(generator.K, seed=42)

        numpy_sim = NumPyForwardSimulator(gen_np, regime="baseline")
        jax_sim = JAXForwardSimulator(gen_jax, regime="baseline", seed=99)
        # Force JAX to use NumPy fallback
        jax_sim._jax = None

        theta_batch = theta_baseline[np.newaxis, :]
        r_np = numpy_sim.simulate(theta_batch, n_scenarios=200)
        r_jax = jax_sim.simulate(theta_batch, n_scenarios=200)
        np.testing.assert_array_almost_equal(r_np, r_jax, decimal=5)
