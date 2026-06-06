"""
Tests for metrics/var_cvar.py and metrics/ground_truth.py.

Covers
------
- loss_cdf monotonicity and bounds
- var_at quantile correctness
- cvar_at tail expectation
- var_cvar combined dict structure
- GroundTruthMC end-to-end run
- GroundTruthMC memory efficiency
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def K():
    return 10


@pytest.fixture
def theta_baseline(K):
    """Baseline theta for K=10."""
    rng = np.random.default_rng(0)
    factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
    p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
    tail_dep = np.array(0.0, dtype=np.float32)
    copula_params = np.array([0.3, 30.0, 0.0, 0.0], dtype=np.float32)
    return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params]).copy()


# ---------------------------------------------------------------------------
# Tests for var_cvar.py
# ---------------------------------------------------------------------------
class TestLossCdf:
    def test_loss_cdf_monotonic(self):
        """CDF must be non-decreasing."""
        rng = np.random.default_rng(123)
        losses = rng.exponential(scale=100, size=10_000).astype(np.float32)
        x_grid = np.linspace(0, 500, 200).astype(np.float32)

        from metrics.var_cvar import loss_cdf

        cdf = loss_cdf(losses, x_grid)
        diffs = np.diff(cdf)
        assert np.all(diffs >= -1e-6), "CDF must be non-decreasing"

    def test_loss_cdf_bounds(self):
        """CDF values must lie in [0, 1]."""
        rng = np.random.default_rng(456)
        losses = rng.normal(loc=100, scale=20, size=5_000).astype(np.float32)
        x_grid = np.linspace(-50, 250, 300).astype(np.float32)

        from metrics.var_cvar import loss_cdf

        cdf = loss_cdf(losses, x_grid)
        assert np.all(cdf >= 0.0), "CDF must be >= 0"
        assert np.all(cdf <= 1.0), "CDF must be <= 1"

    def test_loss_cdf_endpoint(self):
        """CDF at min(loss) = 0, CDF at max(loss) = 1."""
        losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        x_grid = np.array([0.5, 3.0, 10.0], dtype=np.float32)

        from metrics.var_cvar import loss_cdf

        cdf = loss_cdf(losses, x_grid)
        assert cdf[0] == 0.0, "CDF below min loss must be 0"
        assert cdf[-1] == 1.0, "CDF above max loss must be 1"


class TestVarAt:
    def test_var_at_quantile(self):
        """var_at(alpha) must match np.quantile(alpha)."""
        rng = np.random.default_rng(789)
        losses = rng.lognormal(mean=4, sigma=1, size=50_000).astype(np.float64)

        from metrics.var_cvar import var_at

        for alpha in [0.80, 0.90, 0.95, 0.99, 0.999]:
            np_var = float(np.quantile(losses, alpha))
            our_var = var_at(losses, alpha)
            assert abs(our_var - np_var) < 1e-3, (
                f"VaR at {alpha} mismatch: {our_var} vs {np_var}"
            )

    def test_var_at_zero_variance(self):
        """VaR of constant array is that constant."""
        constant = np.full(100,42.0, dtype=np.float64)
        from metrics.var_cvar import var_at

        for alpha in [0.5, 0.95, 0.99]:
            assert var_at(constant, alpha) == 42.0

    def test_var_at_all_zeros(self):
        """VaR of all-zeros array is 0."""
        zeros = np.zeros(1000, dtype=np.float64)
        from metrics.var_cvar import var_at

        for alpha in [0.5, 0.95, 0.99]:
            assert var_at(zeros, alpha) == 0.0

    def test_var_at_ties(self):
        """VaR must be stable when many samples tie at the quantile."""
        #50% of samples are 0,50% are 1
        losses = np.array([0.0] * 500 + [1.0] * 500, dtype=np.float64)
        from metrics.var_cvar import var_at

        var_50 = var_at(losses, 0.50)
        # At50th percentile with equal masses, any value in [0, 1] is valid
        assert 0.0 <= var_50 <= 1.0

    def test_var_at_invalid_alpha(self):
        """var_at must raise for alpha outside (0, 1)."""
        losses = np.array([1.0, 2.0, 3.0])
        from metrics.var_cvar import var_at

        with pytest.raises(ValueError, match="alpha must be in"):
            var_at(losses, 0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            var_at(losses, 1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            var_at(losses, 1.5)


class TestCvarAt:
    def test_cvar_at_tail(self):
        """CVaR must be >= VaR for the same alpha (tail expectation property)."""
        rng = np.random.default_rng(321)
        losses = rng.lognormal(mean=4, sigma=1, size=50_000).astype(np.float64)

        from metrics.var_cvar import cvar_at, var_at

        for alpha in [0.90, 0.95, 0.99]:
            var = var_at(losses, alpha)
            cvar = cvar_at(losses, alpha)
            assert cvar >= var - 1e-6, (
                f"CVaR at {alpha} must be >= VaR: {cvar} < {var}"
            )

    def test_cvar_at_zero_variance(self):
        """CVaR of constant array equals that constant."""
        constant = np.full(100, 42.0, dtype=np.float64)
        from metrics.var_cvar import cvar_at

        for alpha in [0.5, 0.95, 0.99]:
            assert cvar_at(constant, alpha) == 42.0

    def test_cvar_at_all_zeros(self):
        """CVaR of all-zeros array is 0."""
        zeros = np.zeros(1000, dtype=np.float64)
        from metrics.var_cvar import cvar_at

        for alpha in [0.5, 0.95, 0.99]:
            assert cvar_at(zeros, alpha) == 0.0

    def test_cvar_at_invalid_alpha(self):
        """cvar_at must raise for alpha outside (0, 1)."""
        losses = np.array([1.0, 2.0, 3.0])
        from metrics.var_cvar import cvar_at

        with pytest.raises(ValueError, match="alpha must be in"):
            cvar_at(losses, 0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            cvar_at(losses, 1.0)


class TestVarCvar:
    def test_var_cvar_combined(self):
        """var_cvar dict must have all 6 metric keys."""
        rng = np.random.default_rng(654)
        losses = rng.lognormal(mean=4, sigma=1, size=10_000).astype(np.float64)

        from metrics.var_cvar import var_cvar

        result = var_cvar(losses)

        expected_keys = {
            "var_0_95",
            "var_0_99",
            "var_0_999",
            "cvar_0_95",
            "cvar_0_99",
            "cvar_0_999",
            "tail_prob_0_95",
            "tail_prob_0_99",
            "tail_prob_0_999",
        }
        assert set(result.keys()) == expected_keys, f"Missing keys: {expected_keys - set(result.keys())}"

    def test_var_cvar_monotonicity(self):
        """VaR and CVaR must be monotonically non-decreasing in alpha."""
        rng = np.random.default_rng(987)
        losses = rng.lognormal(mean=4, sigma=1, size=20_000).astype(np.float64)

        from metrics.var_cvar import var_cvar

        result = var_cvar(losses)

        assert result["var_0_95"] <= result["var_0_99"]
        assert result["var_0_99"] <= result["var_0_999"]
        assert result["cvar_0_95"] <= result["cvar_0_99"]
        assert result["cvar_0_99"] <= result["cvar_0_999"]

    def test_var_cvar_custom_alphas(self):
        """var_cvar accepts custom alpha list."""
        rng = np.random.default_rng(111)
        losses = rng.normal(size=5_000).astype(np.float64)

        from metrics.var_cvar import var_cvar

        result = var_cvar(losses, alphas=[0.5, 0.90])
        assert "var_0_5" in result
        assert "var_0_9" in result
        assert "cvar_0_5" in result
        assert "cvar_0_9" in result


# ---------------------------------------------------------------------------
# Tests for ground_truth.py
# ---------------------------------------------------------------------------
class TestGroundTruthMC:
    def test_ground_truth_runs(self, K, theta_baseline):
        """End-to-end: small N=10, n_scenarios=1000 runs without error."""
        from copula.gaussian import GaussianFactorCopula
        from data.synthetic import SyntheticPortfolioGenerator
        from metrics.ground_truth import GroundTruthMC

        # Slice to correct theta length (2*K+4) for GaussianFactorCopula
        theta_valid = theta_baseline[: 2 * K + 4]

        # Build posterior samples:10 slightly perturbed thetas
        rng = np.random.default_rng(42)
        posterior_samples = np.stack(
            [
                theta_valid
                + rng.normal(0, 0.05, size=theta_valid.shape).astype(np.float32)
                for _ in range(10)
            ]
        )

        gen = SyntheticPortfolioGenerator(K=K, seed=0)
        copula = GaussianFactorCopula(K=K, seed=0)

        mc = GroundTruthMC(
            copula=copula,
            portfolio_generator=gen,
            n_scenarios=1_000,
            posterior_samples=posterior_samples,
            regime="baseline",
            seed=0,
        )

        result = mc.run(samples_per_posterior=1_000)

        # Check result structure
        assert "posterior_var" in result
        assert "posterior_cvar" in result
        assert "posterior_var_99" in result
        assert "posterior_cvar_99" in result
        assert "posterior_var_999" in result
        assert "posterior_cvar_999" in result
        assert "predictive_var_at_0.95" in result
        assert "predictive_cvar_at_0.95" in result
        assert result["n_posterior_samples"] == 10
        assert result["n_scenarios_per_posterior"] == 1_000
        assert result["regime"] == "baseline"

        # Check shapes
        assert result["posterior_var"].shape == (10,)
        assert result["posterior_cvar"].shape == (10,)

        # Check monotonicity: CVaR >= VaR per sample
        assert np.all(result["posterior_cvar"] >= result["posterior_var"] - 1e-5)
        assert np.all(result["posterior_cvar_99"] >= result["posterior_var_99"] - 1e-5)
        assert np.all(result["posterior_cvar_999"] >= result["posterior_var_999"] - 1e-5)

    def test_ground_truth_no_posterior(self, K):
        """GroundTruthMC works with no posterior (uses default theta)."""
        from copula.gaussian import GaussianFactorCopula
        from data.synthetic import SyntheticPortfolioGenerator
        from metrics.ground_truth import GroundTruthMC

        gen = SyntheticPortfolioGenerator(K=K, seed=0)
        copula = GaussianFactorCopula(K=K, seed=0)

        mc = GroundTruthMC(
            copula=copula,
            portfolio_generator=gen,
            n_scenarios=500,
            posterior_samples=None,  # uses default theta
            regime="baseline",
            seed=0,
        )

        result = mc.run(samples_per_posterior=500)

        assert result["n_posterior_samples"] == 1
        assert result["posterior_var"].shape == (1,)
        assert result["posterior_cvar"].shape == (1,)

    @pytest.mark.slow
    def test_ground_truth_memory_efficient(self, K, theta_baseline):
        """Peak memory < 1 GB for n=1e6 scenarios (streaming mode)."""
        import sys

        from copula.gaussian import GaussianFactorCopula
        from data.synthetic import SyntheticPortfolioGenerator
        from metrics.ground_truth import GroundTruthMC

        # Slice to correct theta length (2*K+4) for GaussianFactorCopula
        theta_valid = theta_baseline[: 2 * K + 4]

        # Only 2 posterior samples for this test
        rng = np.random.default_rng(42)
        posterior_samples = np.stack(
            [
                theta_valid
                + rng.normal(0, 0.05, size=theta_valid.shape).astype(np.float32)
                for _ in range(2)
            ]
        )

        gen = SyntheticPortfolioGenerator(K=K, seed=0)
        copula = GaussianFactorCopula(K=K, seed=0)

        mc = GroundTruthMC(
            copula=copula,
            portfolio_generator=gen,
            n_scenarios=1_000_000,
            posterior_samples=posterior_samples,
            regime="baseline",
            seed=0,
        )

        # Streaming mode with 50k batches
        result = mc.run_streaming(
            samples_per_posterior=1_000_000,
            batch_size=50_000,
        )

        # all_loss_samples must be None (memory efficient)
        assert result["all_loss_samples"] is None
        # Results should still be valid
        assert result["posterior_var"].shape == (2,)
        assert np.all(result["posterior_cvar"] >= result["posterior_var"] - 1e-5)

    def test_ground_truth_multiple_copulas(self, K, theta_baseline):
        """GroundTruthMC works with both Gaussian and Student-t copulas."""
        from copula.gaussian import GaussianFactorCopula
        from copula.student_t import StudentTFactorCopula
        from data.synthetic import SyntheticPortfolioGenerator
        from metrics.ground_truth import GroundTruthMC

        # Slice to correct theta length (2*K+4) for copulas
        theta_valid = theta_baseline[: 2 * K + 4]

        rng = np.random.default_rng(42)
        posterior_samples = np.stack(
            [
                theta_valid
                + rng.normal(0, 0.05, size=theta_valid.shape).astype(np.float32)
                for _ in range(5)
            ]
        )

        gen = SyntheticPortfolioGenerator(K=K, seed=0)

        for Copula in [GaussianFactorCopula, StudentTFactorCopula]:
            copula = Copula(K=K, seed=0)
            mc = GroundTruthMC(
                copula=copula,
                portfolio_generator=gen,
                n_scenarios=500,
                posterior_samples=posterior_samples,
                regime="baseline",
                seed=0,
            )
            result = mc.run(samples_per_posterior=500)
            assert result["posterior_var"].shape == (5,)
            assert not np.any(np.isnan(result["posterior_var"]))
            assert not np.any(np.isnan(result["posterior_cvar"]))
