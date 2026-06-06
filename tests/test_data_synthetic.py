"""
Tests for data/synthetic.py — SyntheticPortfolioGenerator and PortfolioDataset.
"""
from __future__ import annotations

import numpy as np
import pytest
from data.synthetic import (
    SyntheticPortfolioGenerator,
    PortfolioDataset,
    REGIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def K():
    return 10


@pytest.fixture
def generator(K):
    """Default K=10 generator with fixed seed."""
    return SyntheticPortfolioGenerator(K=K, seed=42)


@pytest.fixture
def theta_baseline(K):
    """
    Baseline theta vector for K=10.

    Layout: [factor_loadings(K), p_zeros(K), tail_dep(1), copula_params(3)]
    D = 2*K + 4 = 24
    """
    rng = np.random.default_rng(0)
    factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
    p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
    tail_dep = np.array(0.0, dtype=np.float32)
    copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)  # rho, nu, rotation
    return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])


@pytest.fixture
def theta_tcopula(K):
    """Theta with Student-t tail dependence."""
    rng = np.random.default_rng(1)
    factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
    p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
    tail_dep = np.array(1.0, dtype=np.float32)
    copula_params = np.array([0.4, 5.0, 0.0], dtype=np.float32)
    return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])


# ---------------------------------------------------------------------------
# SyntheticPortfolioGenerator tests
# ---------------------------------------------------------------------------
class TestSyntheticPortfolioGeneratorInit:
    """Tests for __init__."""

    def test_default_K(self):
        gen = SyntheticPortfolioGenerator()
        assert gen.K == 10

    def test_custom_K(self):
        gen = SyntheticPortfolioGenerator(K=5)
        assert gen.K == 5

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            SyntheticPortfolioGenerator(K=0)

    def test_seed_reproducibility(self, K):
        """Same seed + same theta → identical losses."""
        rng = np.random.default_rng(0)
        theta_fixed = rng.uniform(-0.5, 0.5, size=2 * K + 4).astype(np.float32)
        gen1 = SyntheticPortfolioGenerator(K=K, seed=99)
        gen2 = SyntheticPortfolioGenerator(K=K, seed=99)
        ds1 = gen1.sample(theta_fixed, n_scenarios=100)
        ds2 = gen2.sample(theta_fixed, n_scenarios=100)
        np.testing.assert_array_equal(ds1.losses, ds2.losses)


class TestSyntheticPortfolioGeneratorSample:
    """Tests for sample()."""

    def test_output_type(self, generator, theta_baseline):
        ds = generator.sample(theta_baseline, n_scenarios=500)
        assert isinstance(ds, PortfolioDataset)

    def test_losses_shape(self, generator, theta_baseline):
        n_scenarios = 500
        ds = generator.sample(theta_baseline, n_scenarios=n_scenarios)
        assert ds.losses.shape == (n_scenarios,)
        assert ds.losses.dtype == np.float32

    def test_observations_shape(self, generator, theta_baseline):
        n_scenarios = 500
        ds = generator.sample(theta_baseline, n_scenarios=n_scenarios)
        assert ds.observations.shape == (10,)
        assert ds.observations.dtype == np.float32

    def test_theta_preserved(self, generator, theta_baseline):
        ds = generator.sample(theta_baseline, n_scenarios=100)
        np.testing.assert_array_equal(ds.theta, theta_baseline)

    def test_theta_wrong_length_raises(self, generator):
        wrong_theta = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="theta must have shape"):
            generator.sample(wrong_theta, n_scenarios=100)

    def test_losses_non_negative(self, generator, theta_baseline):
        ds = generator.sample(theta_baseline, n_scenarios=1000)
        assert np.all(ds.losses >= 0.0)

    def test_observations_columns_10(self, generator, theta_baseline):
        ds = generator.sample(theta_baseline, n_scenarios=100)
        assert ds.observations.shape == (10,)

    def test_var95_within_loss_range(self, generator, theta_baseline):
        ds = generator.sample(theta_baseline, n_scenarios=1000)
        var95 = ds.observations[9]
        assert var95 >= 0.0
        assert var95 <= ds.losses.max() * 1.5


class TestLossesToObservation:
    """Tests for losses_to_observation()."""

    def test_output_shape(self, generator, theta_baseline):
        generator.sample(theta_baseline, n_scenarios=100)
        losses = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0], dtype=np.float32)
        obs = generator.losses_to_observation(losses)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32

    def test_var95_column(self, generator, theta_baseline):
        generator.sample(theta_baseline, n_scenarios=100)
        losses = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0], dtype=np.float32)
        obs = generator.losses_to_observation(losses)
        # np.percentile with linear interpolation gives1800.0 for numpy 2.x
        assert obs[9] == 1800.0

    def test_n_defaults_heuristic_bounded(self, generator, theta_baseline):
        generator.sample(theta_baseline, n_scenarios=100)
        losses = np.array([0.0, 1e6, 2e6], dtype=np.float32)
        obs = generator.losses_to_observation(losses)
        assert obs[0] >= 0
        assert obs[0] <= generator.K


class TestStudentTCopula:
    """Tests for Student-t copula branch (tail_dep > 0)."""

    def test_sample_runs(self, generator, theta_tcopula):
        ds = generator.sample(theta_tcopula, n_scenarios=200)
        assert ds.losses.shape == (200,)
        assert ds.observations.shape == (10,)

    def test_losses_non_negative_tcopula(self, generator, theta_tcopula):
        ds = generator.sample(theta_tcopula, n_scenarios=500)
        assert np.all(ds.losses >= 0.0)


class TestEdgeCases:
    """Edge case tests."""

    def test_K1(self):
        gen = SyntheticPortfolioGenerator(K=1, seed=0)
        rng = np.random.default_rng(0)
        theta = np.concatenate([
            rng.uniform(-0.5, 0.5, size=1).astype(np.float32),
            rng.uniform(0.005, 0.10, size=1).astype(np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.3, 30.0, 0.0], dtype=np.float32),
        ])
        ds = gen.sample(theta, n_scenarios=50)
        assert ds.losses.shape == (50,)
        assert ds.observations.shape == (10,)

    def test_observations_single_scenario(self, generator, theta_baseline):
        ds = generator.sample(theta_baseline, n_scenarios=1)
        assert ds.observations.shape == (10,)
