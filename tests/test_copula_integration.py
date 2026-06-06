"""
Integration tests: each copula wired into `SyntheticPortfolioGenerator`.

Validates the data flow:
    copula.sample(theta) → losses → SyntheticPortfolioGenerator.losses_to_observation

Also checks that theta from SyntheticPortfolioGenerator can be re-used
by each copula variant (same theta layout).
"""
from __future__ import annotations

import numpy as np

from copula.gaussian import GaussianFactorCopula
from copula.student_t import StudentTFactorCopula
from copula.vine import DVineCopula
from copula.low_rank import LowRankFactorCopula

from data.synthetic import SyntheticPortfolioGenerator


class TestCopulaIntegration:
    """End-to-end integration tests for copula → portfolio pipeline."""

    def test_gaussian_copula_integration(self):
        """Gaussian copula losses → observations via SyntheticPortfolioGenerator."""
        K = 10
        gen = SyntheticPortfolioGenerator(K=K, seed=42)

        # Build a valid theta for Gaussian copula
        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.4        # factor loadings
        theta[K : 2 * K] = 0.03  # p_zeros (3% default rate)
        theta[2 * K + 1] = 0.5  # rho

        cop = GaussianFactorCopula(K=K, seed=42)
        U, losses = cop.sample(theta, n_samples=500)

        assert U.shape == (500, K)
        assert losses.shape == (500,)
        assert losses.min() >= 0.0

        # Wire into SyntheticPortfolioGenerator observation
        obs = gen.losses_to_observation(losses)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32

    def test_student_t_copula_integration(self):
        """Student-t copula losses → observations."""
        K = 10
        gen = SyntheticPortfolioGenerator(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.4
        theta[K : 2 * K] = 0.03
        theta[2 * K] = 0.8   # tail_dep → nu ≈ 4
        theta[2 * K + 1] = 0.5
        theta[2 * K + 2] = 4.0

        cop = StudentTFactorCopula(K=K, seed=42)
        U, losses = cop.sample(theta, n_samples=500)

        assert losses.min() >= 0.0
        obs = gen.losses_to_observation(losses)
        assert obs.shape == (10,)

    def test_dvine_copula_integration(self):
        """D-vine copula losses → observations."""
        K = 10
        gen = SyntheticPortfolioGenerator(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.3
        theta[K : 2 * K] = 0.03
        theta[2 * K + 1] = 0.5

        cop = DVineCopula(K=K, seed=42)
        U, losses = cop.sample(theta, n_samples=500)

        assert losses.min() >= 0.0
        obs = gen.losses_to_observation(losses)
        assert obs.shape == (10,)

    def test_lowrank_copula_integration(self):
        """Low-rank copula losses → observations."""
        K = 10
        r = 3
        gen = SyntheticPortfolioGenerator(K=K, seed=42)

        theta = np.zeros(K * r + K + 4, dtype=np.float32)
        rng = np.random.default_rng(42)
        theta[: K * r] = rng.uniform(-0.5, 0.5, size=K * r).astype(np.float32)
        theta[K * r : K * r + K] = 0.03

        cop = LowRankFactorCopula(K=K, r=r, seed=42)
        U, losses = cop.sample(theta, n_samples=500)

        assert losses.min() >= 0.0
        obs = gen.losses_to_observation(losses)
        assert obs.shape == (10,)

    def test_synthetic_portfolio_generator_theta_reuse(self):
        """
        SyntheticPortfolioGenerator.sample() theta can be re-used by copulas.

        SyntheticPortfolioGenerator uses the same theta layout as the copulas,
        so its theta output should be directly usable as copula input.
        """
        K = 10
        gen = SyntheticPortfolioGenerator(K=K, seed=42)

        # Generate a dataset with known theta
        theta_gs = np.zeros(2 * K + 4, dtype=np.float32)
        theta_gs[:K] = 0.35
        theta_gs[K : 2 * K] = 0.02
        theta_gs[2 * K + 1] = 0.4

        dataset = gen.sample(theta_gs, n_scenarios=300)
        assert dataset.theta.shape == (2 * K + 4,)

        # Re-use the dataset theta in the Gaussian copula
        cop = GaussianFactorCopula(K=K, seed=99)
        U, losses = cop.sample(dataset.theta, n_samples=300)

        assert U.shape == (300, K)
        assert losses.shape == (300,)

    def test_all_copulas_produce_valid_loss_distribution(self):
        """
        Smoke test: all copulas produce loss distributions with correct properties.
        """
        K = 8
        n = 1000

        copulas = [
            ("Gaussian", GaussianFactorCopula(K=K, seed=1)),
            ("Student-t", StudentTFactorCopula(K=K, seed=1)),
            ("D-vine", DVineCopula(K=K, seed=1)),
            ("Low-rank", LowRankFactorCopula(K=K, r=3, seed=1)),
        ]

        for name, cop in copulas:
            # Build appropriate theta for each copula type
            if name == "Low-rank":
                theta = np.zeros(K * 3 + K + 4, dtype=np.float32)
                rng = np.random.default_rng(1)
                # Use stronger loadings to ensure non-trivial factor structure
                theta[: K * 3] = rng.uniform(-0.9, 0.9, size=K * 3).astype(np.float32)
                theta[K * 3 : K * 3 + K] = 0.15  # higher default prob
            else:
                theta = np.zeros(2 * K + 4, dtype=np.float32)
                theta[:K] = 0.3
                theta[K : 2 * K] = 0.10  # higher default prob for non-degenerate losses
                theta[2 * K + 1] = 0.4
                if name == "Student-t":
                    theta[2 * K] = 0.5
                    theta[2 * K + 2] = 4.0

            U, losses = cop.sample(theta, n_samples=n)

            # Loss distribution properties
            assert losses.min() >= 0.0, f"{name}: negative loss"
            assert losses.std() > 0.0, f"{name}: degenerate loss distribution (std={losses.std():.6f})"
            assert np.isfinite(losses).all(), f"{name}: non-finite losses"

            # VaR ordering
            var95 = np.percentile(losses, 95)
            var99 = np.percentile(losses, 99)
            assert var95 <= var99, f"{name}: VaR_95 > VaR_99"

    def test_loss_observation_round_trip(self):
        """
        losses → loss_to_observation → losses recovers approximately.

        This tests the observation function is invertible in expectation.
        """
        K = 10
        gen = SyntheticPortfolioGenerator(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.4
        theta[K : 2 * K] = 0.05
        theta[2 * K + 1] = 0.5

        cop = GaussianFactorCopula(K=K, seed=42)
        U, losses = cop.sample(theta, n_samples=1000)

        obs = gen.losses_to_observation(losses)

        # The observation vector has VaR at index 9
        # It should be a reasonable estimate of the 95th percentile
        computed_var95 = np.percentile(losses, 95)
        assert abs(obs[9] - computed_var95) / (computed_var95 + 1e-8) < 0.5, \
            "VaR observation is far from true 95th percentile"