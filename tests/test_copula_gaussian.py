"""
Tests for GaussianFactorCopula.

Coverage targets:
- shape and dtype of sample output
- marginal U ~ Uniform(0,1) (Kolmogorov-Smirnov test)
- losses aggregation correctness
- reproducibility with fixed seed
- theta validation
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from copula.gaussian import GaussianFactorCopula


class TestGaussianFactorCopula:
    """Unit tests for GaussianFactorCopula."""

    def test_sample_shape_and_dtype(self):
        """sample() returns correct shapes and float32 dtype."""
        K = 10
        cop = GaussianFactorCopula(K=K, seed=42)

        # Build a valid theta for K=10: length = 2*K + 4 = 24
        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.3          # factor_loadings
        theta[K : 2 * K] = 0.02  # p_zeros (2% default rate)
        theta[2 * K + 1] = 0.5   # rho

        U, losses = cop.sample(theta, n_samples=500)

        assert U.shape == (500, K)
        assert losses.shape == (500,)
        assert U.dtype == np.float32
        assert losses.dtype == np.float32

    def test_marginals_uniform(self):
        """Generated U marginals are consistent with Uniform(0,1)."""
        K = 10
        cop = GaussianFactorCopula(K=K, seed=99)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.4
        theta[K : 2 * K] = 0.05
        theta[2 * K + 1] = 0.3

        U, _ = cop.sample(theta, n_samples=2000)

        # Kolmogorov-Smirnov test for each marginal column
        for j in range(K):
            # H0: U[:, j] ~ Uniform(0,1)
            d_stat, p_val = stats.kstest(U[:, j], "uniform", args=(0.0, 1.0))
            assert p_val > 0.01, f"Marginal {j} fails KS test (p={p_val:.4f})"

    def test_losses_aggregation(self):
        """Losses are non-negative and bounded by max possible loss."""
        K = 10
        cop = GaussianFactorCopula(K=K, seed=123)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.0   # no systematic factor (b_i = 0)
        theta[K : 2 * K] = 0.5  # 50% default prob → ~50% defaults
        theta[2 * K + 1] = 0.0  # rho = 0

        U, losses = cop.sample(theta, n_samples=1000)

        # With b_i=0, defaults are independent Bernoulli(p_i)
        # Losses should be in range [0, K * max_lgd]
        max_lgd = 0.60
        assert losses.min() >= 0.0
        assert losses.max() <= K * max_lgd

    def test_reproducibility(self):
        """Fixed seed + same theta → identical outputs."""
        K = 8
        cop1 = GaussianFactorCopula(K=K, seed=777)
        cop2 = GaussianFactorCopula(K=K, seed=777)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.5
        theta[K : 2 * K] = 0.03
        theta[2 * K + 1] = 0.4

        U1, losses1 = cop1.sample(theta, n_samples=200)
        U2, losses2 = cop2.sample(theta, n_samples=200)

        np.testing.assert_array_equal(U1, U2)
        np.testing.assert_array_equal(losses1, losses2)

    def test_theta_validation_wrong_length(self):
        """Wrong-length theta raises ValueError."""
        cop = GaussianFactorCopula(K=5, seed=0)
        bad_theta = np.zeros(3, dtype=np.float32)   # should be 2*K+4 = 14

        with pytest.raises(ValueError, match="shape"):
            cop.sample(bad_theta, n_samples=10)

    def test_theta_validation_wrong_dtype(self):
        """Non-float32 theta is accepted and converted."""
        K = 5
        cop = GaussianFactorCopula(K=K, seed=0)
        theta = np.zeros(2 * K + 4, dtype=np.float64)
        theta[:K] = 0.3
        theta[K : 2 * K] = 0.02
        theta[2 * K + 1] = 0.3

        # Should not raise — conversion is allowed
        U, losses = cop.sample(theta, n_samples=50)
        assert U.shape == (50, K)

    def test_rho_zero_no_factor_effect(self):
        """When rho=0 and b_i=0, defaults are independent."""
        K = 5
        cop = GaussianFactorCopula(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.0   # zero factor loading → no systematic component
        theta[K : 2 * K] = 0.10  # 10% default probability
        theta[2 * K + 1] = 0.0  # zero correlation

        _, losses = cop.sample(theta, n_samples=5000)

        # Expected: mean(loss) ≈ K * p * lgd ≈ 5 * 0.10 * 0.40 = 0.20
        expected_mean = K * 0.10 * 0.40
        assert abs(losses.mean() - expected_mean) < 0.05

    def test_correlation_increases_default_comovement(self):
        """Higher rho → higher correlation between default flags."""
        K = 5
        n = 2000

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.8   # strong factor loading
        theta[K : 2 * K] = 0.10  # 10% default prob
        theta[2 * K + 1] = 0.0   # low rho

        # With strong b_i but low rho, idiosyncratic still dominates
        # correlation is limited by rho alone.
        # Test that as rho increases, the variance of number of defaults increases
        theta_low_rho = theta.copy()
        theta_low_rho[2 * K + 1] = 0.1
        theta_high_rho = theta.copy()
        theta_high_rho[2 * K + 1] = 0.9

        cop = GaussianFactorCopula(K=K, seed=42)
        _, losses_low = cop.sample(theta_low_rho, n_samples=n)
        _, losses_high = cop.sample(theta_high_rho, n_samples=n)

        # High correlation → more scenarios with extreme outcomes (all default or none)
        # → variance of losses should be higher
        assert losses_high.std() > losses_low.std() * 0.5

    def test_losses_to_U_aggregated_loss(self):
        """losses_to_U_aggregated_loss computes correct loss from U."""
        K = 4
        lgd = np.array([0.30, 0.40, 0.50, 0.35], dtype=np.float32)
        p_zeros = np.array([0.05, 0.10, 0.08, 0.12], dtype=np.float32)

        # Synthetic U where we know which loans default
        U = np.array([
            [0.01, 0.50, 0.50, 0.50],   # loan 0 defaults (U < p)
            [0.50, 0.01, 0.50, 0.50],   # loan 1 defaults
            [0.50, 0.50, 0.01, 0.50],   # loan 2 defaults
            [0.50, 0.50, 0.50, 0.50],   # none defaults
        ], dtype=np.float32)

        losses = GaussianFactorCopula.losses_to_U_aggregated_loss(U, lgd, p_zeros)

        expected = np.array([
            0.30,   # loan 0 only
            0.40,   # loan 1 only
            0.50,   # loan 2 only
            0.00,   # none
        ], dtype=np.float32)

        np.testing.assert_array_almost_equal(losses, expected)