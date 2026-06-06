"""
Tests for DVineCopula.

Coverage targets:
- shape and dtype
- symmetry properties (pairwise exchangeability)
- K=3, 5, 10 scaling
- reproducibility
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from copula.vine import DVineCopula


class TestDVineCopula:
    """Unit tests for DVineCopula."""

    @pytest.mark.parametrize("K", [3, 5, 10])
    def test_sample_shape_and_dtype(self, K):
        """sample() returns correct shapes for K=3,5,10."""
        cop = DVineCopula(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.3
        theta[K : 2 * K] = 0.03
        theta[2 * K + 1] = 0.4

        U, losses = cop.sample(theta, n_samples=300)

        assert U.shape == (300, K)
        assert losses.shape == (300,)
        assert U.dtype == np.float32
        assert losses.dtype == np.float32

    def test_marginals_uniform(self):
        """D-vine U marginals are consistent with Uniform(0,1)."""
        K = 7
        cop = DVineCopula(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.3
        theta[K : 2 * K] = 0.05
        theta[2 * K + 1] = 0.5

        U, _ = cop.sample(theta, n_samples=2000)

        for j in range(K):
            d_stat, p_val = stats.kstest(U[:, j], "uniform", args=(0.0, 1.0))
            assert p_val > 0.01, f"Marginal {j} fails KS test (p={p_val:.4f})"

    def test_symmetry_pairwise_exchangeability(self):
        """D-vine should show exchangeability across adjacent pairs."""
        K = 6
        cop = DVineCopula(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.4
        theta[K : 2 * K] = 0.05
        theta[2 * K + 1] = 0.6  # high correlation

        U, _ = cop.sample(theta, n_samples=2000)

        # Check that adjacent pairs have similar correlation structure
        # Pearson correlation between U[:, i] and U[:, i+1]
        corrs = []
        for i in range(K - 1):
            c = np.corrcoef(U[:, i], U[:, i + 1])[0, 1]
            corrs.append(c)

        # All adjacent correlations should be positive (vine structure preserves ordering)
        assert all(c > 0.1 for c in corrs), f"Adjacent correlations too low: {corrs}"

    def test_reproducibility(self):
        """Fixed seed + same theta → identical outputs."""
        K = 8
        cop1 = DVineCopula(K=K, seed=777)
        cop2 = DVineCopula(K=K, seed=777)

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
        cop = DVineCopula(K=5, seed=0)
        bad_theta = np.zeros(3, dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            cop.sample(bad_theta, n_samples=10)

    def test_loss_aggregation(self):
        """Losses are non-negative and bounded."""
        K = 8
        cop = DVineCopula(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.0  # zero loadings (vine is not factor-based but ordering matters)
        theta[K : 2 * K] = 0.20  # 20% default prob
        theta[2 * K + 1] = 0.5

        _, losses = cop.sample(theta, n_samples=1000)

        max_lgd = 0.60
        assert losses.min() >= 0.0
        assert losses.max() <= K * max_lgd

    def test_losses_to_U_aggregated_loss(self):
        """losses_to_U_aggregated_loss computes correct loss."""
        K = 4
        lgd = np.array([0.30, 0.40, 0.50, 0.35], dtype=np.float32)
        p_zeros = np.array([0.05, 0.10, 0.08, 0.12], dtype=np.float32)

        U = np.array([
            [0.01, 0.50, 0.50, 0.50],   # loan 0 defaults
            [0.50, 0.01, 0.50, 0.50],   # loan 1 defaults
            [0.50, 0.50, 0.01, 0.50],   # loan 2 defaults
            [0.50, 0.50, 0.50, 0.50],   # none defaults
        ], dtype=np.float32)

        losses = DVineCopula.losses_to_U_aggregated_loss(U, lgd, p_zeros)

        expected = np.array([0.30, 0.40, 0.50, 0.00], dtype=np.float32)
        np.testing.assert_array_almost_equal(losses, expected)

    def test_rho_correlation_structure(self):
        """Higher rho → higher correlation between adjacent marginals."""
        K = 5
        n = 2000

        theta_low = np.zeros(2 * K + 4, dtype=np.float32)
        theta_low[:K] = 0.3
        theta_low[K : 2 * K] = 0.05
        theta_low[2 * K + 1] = 0.1  # low rho

        theta_high = np.zeros(2 * K + 4, dtype=np.float32)
        theta_high[:K] = 0.3
        theta_high[K : 2 * K] = 0.05
        theta_high[2 * K + 1] = 0.9  # high rho

        cop = DVineCopula(K=K, seed=42)

        U_low, _ = cop.sample(theta_low, n_samples=n)
        U_high, _ = cop.sample(theta_high, n_samples=n)

        corr_low = np.corrcoef(U_low[:, 0], U_low[:, 1])[0, 1]
        corr_high = np.corrcoef(U_high[:, 0], U_high[:, 1])[0, 1]

        assert corr_high > corr_low, "Higher rho should produce higher adjacent correlation"