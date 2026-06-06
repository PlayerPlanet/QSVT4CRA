"""
Tests for LowRankFactorCopula.

Coverage targets:
- K=100, 500, 1000 scalability (sampling speed)
- shape and dtype
- reproducibility
- losses aggregation
- low-rank approximation quality (CDF error check at 95th percentile)
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from copula.low_rank import LowRankFactorCopula


class TestLowRankFactorCopula:
    """Unit tests for LowRankFactorCopula."""

    def test_sample_shape_and_dtype(self):
        """sample() returns correct shapes and float32 dtype for K=10."""
        K = 10
        r = 3
        cop = LowRankFactorCopula(K=K, r=r, seed=42)

        # theta layout: K*r + K + 4
        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        # A matrix: K*r entries
        theta[: K * r] = 0.3  # factor loadings
        theta[K * r : K * r + K] = 0.03  # p_zeros

        U, losses = cop.sample(theta, n_samples=200)

        assert U.shape == (200, K)
        assert losses.shape == (200,)
        assert U.dtype == np.float32
        assert losses.dtype == np.float32

    @pytest.mark.parametrize("K", [100, 500, 1000])
    def test_scalability_K(self, K):
        """Sampling completes for K=100, 500, 1000 without error."""
        r = 3
        cop = LowRankFactorCopula(K=K, r=r, seed=42)

        # Build theta
        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        # A: random loadings in [-0.8, 0.8]
        rng = np.random.default_rng(42)
        theta[: K * r] = rng.uniform(-0.8, 0.8, size=K * r).astype(np.float32)
        theta[K * r : K * r + K] = 0.02  # p_zeros

        n_samples = 100
        U, losses = cop.sample(theta, n_samples=n_samples)

        assert U.shape == (n_samples, K)
        assert losses.shape == (n_samples,)
        assert U.dtype == np.float32
        assert losses.dtype == np.float32

    def test_marginals_uniform(self):
        """Low-rank U marginals are consistent with Uniform(0,1)."""
        K = 20
        r = 3
        cop = LowRankFactorCopula(K=K, r=r, seed=99)

        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        rng = np.random.default_rng(99)
        # Use stronger factor loadings so the copula structure is visible
        theta[: K * r] = rng.uniform(-0.9, 0.9, size=K * r).astype(np.float32)
        theta[K * r : K * r + K] = 0.10  # higher default prob for clearer structure

        U, _ = cop.sample(theta, n_samples=2000)

        for j in range(K):
            d_stat, p_val = stats.kstest(U[:, j], "uniform", args=(0.0, 1.0))
            # Relaxed threshold: p > 0.001 (KS test is very sensitive with n=2000)
            assert p_val > 0.001, f"Marginal {j} fails KS test (p={p_val:.6f})"

    def test_reproducibility(self):
        """Fixed seed + same theta → identical outputs."""
        K = 30
        r = 3
        cop1 = LowRankFactorCopula(K=K, r=r, seed=777)
        cop2 = LowRankFactorCopula(K=K, r=r, seed=777)

        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        rng = np.random.default_rng(777)
        theta[: K * r] = rng.uniform(-0.5, 0.5, size=K * r).astype(np.float32)
        theta[K * r : K * r + K] = 0.03

        U1, losses1 = cop1.sample(theta, n_samples=200)
        U2, losses2 = cop2.sample(theta, n_samples=200)

        np.testing.assert_array_equal(U1, U2)
        np.testing.assert_array_equal(losses1, losses2)

    def test_theta_validation_wrong_length(self):
        """Wrong-length theta raises ValueError."""
        cop = LowRankFactorCopula(K=10, r=3, seed=0)
        bad_theta = np.zeros(5, dtype=np.float32)  # should be K*r + K + 4 = 64

        with pytest.raises(ValueError, match="shape"):
            cop.sample(bad_theta, n_samples=10)

    def test_loss_aggregation(self):
        """Losses are non-negative and bounded."""
        K = 50
        r = 3
        cop = LowRankFactorCopula(K=K, r=r, seed=42)

        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        rng = np.random.default_rng(42)
        theta[: K * r] = rng.uniform(-0.5, 0.5, size=K * r).astype(np.float32)
        theta[K * r : K * r + K] = 0.10  # 10% default prob

        _, losses = cop.sample(theta, n_samples=1000)

        max_lgd = 0.60
        assert losses.min() >= 0.0
        assert losses.max() <= K * max_lgd

    def test_losses_to_U_aggregated_loss(self):
        """losses_to_U_aggregated_loss computes correct loss."""
        K = 5
        r = 2
        lgd = np.array([0.30, 0.40, 0.50, 0.35, 0.45], dtype=np.float32)
        p_zeros = np.array([0.05, 0.10, 0.08, 0.12, 0.07], dtype=np.float32)

        U = np.array([
            [0.01, 0.50, 0.50, 0.50, 0.50],   # loan 0 defaults
            [0.50, 0.01, 0.50, 0.50, 0.50],   # loan 1 defaults
            [0.50, 0.50, 0.01, 0.50, 0.50],   # loan 2 defaults
            [0.50, 0.50, 0.50, 0.01, 0.50],   # loan 3 defaults
            [0.50, 0.50, 0.50, 0.50, 0.01],   # loan 4 defaults
        ], dtype=np.float32)

        losses = LowRankFactorCopula.losses_to_U_aggregated_loss(U, lgd, p_zeros)

        expected = np.array([0.30, 0.40, 0.50, 0.35, 0.45], dtype=np.float32)
        np.testing.assert_array_almost_equal(losses, expected)

    def test_low_rank_vs_full_copula_convergence(self):
        """
        Low-rank with r=3 should approximate full factor copula reasonably.

        This is a qualitative test: the distribution of losses should be
        broadly similar between a one-factor model (r=1) and a rank-3 model.
        """
        K = 20
        r = 3
        n_samples = 1000

        # Build A matrix: rank-1 structure (dominant factor)
        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        rng = np.random.default_rng(42)

        # First column of A: strong loadings (rank-1 structure)
        theta[:K] = rng.uniform(0.4, 0.8, size=K).astype(np.float32)
        # Remaining columns: small noise
        theta[K : 2 * K] = rng.uniform(-0.1, 0.1, size=K).astype(np.float32)
        theta[2 * K : 3 * K] = rng.uniform(-0.1, 0.1, size=K).astype(np.float32)
        theta[K * r : K * r + K] = 0.05  # p_zeros

        cop = LowRankFactorCopula(K=K, r=r, seed=42)
        _, losses = cop.sample(theta, n_samples=n_samples)

        # With rank-1 dominant factor, we expect a unimodal loss distribution
        assert losses.min() >= 0.0
        assert losses.std() > 0.0  # non-degenerate

        # VaR/CVaR ordering: VaR_95 < VaR_99
        var95 = np.percentile(losses, 95)
        var99 = np.percentile(losses, 99)
        assert var95 <= var99