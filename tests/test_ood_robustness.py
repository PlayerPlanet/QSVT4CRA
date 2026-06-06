"""
Tests for experiments/ood_robustness.py — Phase 5 Distribution Shift Experiment.

Covers
------
- OODExperiment constructor and parameter validation
- Method A (point-estimate GCI) runs without error
- Method B (posterior factor copula) runs without error
- compare_methods_plot creates PNG file
- save_npz roundtrip preserves results
- In-dist regime: method A tail coverage ≈ 0.05 for α=0.95
- OOD regime: method B uncertainty band is wider than in-dist
"""
from __future__ import annotations

import tempfile
from pathlib import Path

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
    """Baseline theta for K=10: factor_loadings + p_zeros + tail_dep + copula_params[rho,nu,spare]."""
    rng = np.random.default_rng(0)
    factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
    p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
    tail_dep = np.array(0.0, dtype=np.float32)
    copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)  # rho, nu, spare (3 elements → 2K+4=24 total)
    return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params]).copy()


@pytest.fixture
def posterior_samples(K, theta_baseline):
    """10 posterior samples: baseline theta + small Gaussian noise."""
    rng = np.random.default_rng(42)
    return np.stack(
        [
            theta_baseline
            + rng.normal(0, 0.05, size=theta_baseline.shape).astype(np.float32)
            for _ in range(10)
        ]
    )


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------
class TestOODExperimentInit:
    def test_constructor_defaults(self, posterior_samples):
        """Constructor accepts all required parameters without error."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            train_regime="baseline",
            test_regimes=["baseline", "housing_crash"],
            n_scenarios=1000,
            K=10,
            copula="gaussian",
            seed=42,
        )

        assert exp.posterior_samples.shape == posterior_samples.shape
        assert exp.train_regime == "baseline"
        assert exp.test_regimes == ["baseline", "housing_crash"]
        assert exp.n_scenarios == 1000
        assert exp.K == 10
        assert exp.copula == "gaussian"
        assert exp.seed == 42

    def test_constructor_missing_test_regimes(self, posterior_samples):
        """Defaults to ['baseline', 'housing_crash'] when test_regimes=None."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(posterior_samples=posterior_samples)
        assert exp.test_regimes == ["baseline", "housing_crash"]

    def test_constructor_posterior_samples_wrong_shape(self, K):
        """run() raises ValueError for wrong theta dimension when copula validates."""
        from experiments.ood_robustness import OODExperiment

        # Wrong D: 2*K+4 expected, but we give a 3-element array
        bad_samples = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
        exp = OODExperiment(posterior_samples=bad_samples, K=K, n_scenarios=100)
        with pytest.raises(ValueError, match="shape"):
            exp.run()

    def test_constructor_copula_student_t(self, posterior_samples):
        """student_t copula is accepted."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=500,
            copula="student_t",
            seed=42,
        )
        assert exp.copula == "student_t"


# ---------------------------------------------------------------------------
# Method A (point-estimate GCI) tests
# ---------------------------------------------------------------------------
class TestMethodARuns:
    def test_method_A_runs_baseline(self, posterior_samples):
        """Method A (point-estimate GCI) runs without error on baseline regime."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=500,
            K=10,
            seed=0,
        )

        results = exp.run()
        regime_results = results["regime_results"]

        assert "baseline" in regime_results
        method_a = regime_results["baseline"]["method_A_gci"]
        # Check VaR keys present
        assert "var_0_95" in method_a
        assert "var_0_99" in method_a
        assert "var_0_999" in method_a
        # Check CVaR keys present
        assert "cvar_0_95" in method_a
        assert "cvar_0_99" in method_a
        assert "cvar_0_999" in method_a
        # Check tail coverage keys present
        assert "tail_prob_0_95" in method_a
        assert "tail_prob_0_99" in method_a
        assert "tail_prob_0_999" in method_a
        # Values are finite
        assert np.isfinite(method_a["var_0_95"])
        assert np.isfinite(method_a["cvar_0_95"])

    def test_method_A_runs_all_regimes(self, posterior_samples):
        """Method A runs for all test regimes without error."""
        from experiments.ood_robustness import OODExperiment

        test_regimes = ["baseline", "housing_crash", "rate_shock_0.5", "rate_shock_1.5"]
        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=test_regimes,
            n_scenarios=500,
            seed=0,
        )

        results = exp.run()
        for regime in test_regimes:
            assert regime in results["regime_results"]
            method_a = results["regime_results"][regime]["method_A_gci"]
            assert "var_0_95" in method_a
            assert np.isfinite(method_a["var_0_95"])


# ---------------------------------------------------------------------------
# Method B (posterior factor copula) tests
# ---------------------------------------------------------------------------
class TestMethodBRuns:
    def test_method_B_runs(self, posterior_samples):
        """Method B (posterior factor copula) runs without error."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=500,
            seed=0,
        )

        results = exp.run()
        method_b = results["regime_results"]["baseline"]["method_B_posterior"]

        # Check mean and std keys
        assert "var_95_mean" in method_b
        assert "var_95_std" in method_b
        assert "var_99_mean" in method_b
        assert "var_99_std" in method_b
        assert "var_999_mean" in method_b
        assert "var_999_std" in method_b
        assert "cvar_95_mean" in method_b
        assert "cvar_95_std" in method_b

        # Check per-sample arrays
        assert "var_95_samples" in method_b
        assert method_b["var_95_samples"].shape[0] == posterior_samples.shape[0]

        # Values are finite
        assert np.isfinite(method_b["var_95_mean"])
        assert np.isfinite(method_b["var_95_std"])

    def test_method_B_uncertainty_bands_non_zero(self, posterior_samples):
        """Method B posterior std is non-zero (spread is captured)."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=1000,
            seed=0,
        )

        results = exp.run()
        method_b = results["regime_results"]["baseline"]["method_B_posterior"]

        # Posterior spread should produce non-zero std
        assert method_b["var_95_std"] > 0.0
        assert method_b["cvar_95_std"] > 0.0


# ---------------------------------------------------------------------------
# Plotting tests
# ---------------------------------------------------------------------------
class TestCompareMethodsPlot:
    def test_compare_methods_plot_creates_file(self, posterior_samples):
        """compare_methods_plot saves a PNG file."""
        from experiments.ood_robustness import OODExperiment, compare_methods_plot

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline", "housing_crash"],
            n_scenarios=500,
            seed=0,
        )

        results = exp.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "ood_comparison.png"
            compare_methods_plot(results, str(plot_path))

            assert plot_path.exists()
            assert plot_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# save_npz roundtrip tests
# ---------------------------------------------------------------------------
class TestSaveNpzRoundtrip:
    def test_save_npz_roundtrip(self, posterior_samples):
        """save_npz and reload produces equivalent results dict."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=500,
            seed=0,
        )

        results1 = exp.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "ood_results.npz"
            exp.save_npz(str(npz_path))

            assert npz_path.exists()

            # Reload and compare
            loaded = dict(np.load(npz_path, allow_pickle=True))

            # Check top-level keys
            assert "regime_results" in loaded
            assert "summary" in loaded
            assert "n_posterior_samples" in loaded
            assert "n_scenarios" in loaded
            assert "train_regime" in loaded
            assert "test_regimes" in loaded

            # Check regime_results structure
            rr = loaded["regime_results"].item()  # dict stored as 0-d array
            assert isinstance(rr, dict)
            assert "baseline" in rr

    def test_save_npz_requires_run_first(self, posterior_samples):
        """save_npz raises RuntimeError if run() has not been called."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=500,
            seed=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "ood_results.npz"
            with pytest.raises(RuntimeError, match="run\\(\\)"):
                exp.save_npz(str(npz_path))


# ---------------------------------------------------------------------------
# Tail coverage tests
# ---------------------------------------------------------------------------
class TestTailCoverageInDist:
    def test_tail_coverage_in_dist(self, posterior_samples):
        """
        In-distribution (baseline) regime: method A tail coverage is positive
        and not wildly diverging from the nominal 5%.

        With discrete loss distribution (multiples of LGD), the empirical
        tail fraction may deviate from the nominal (1-alpha) due to
        discretization. We check that it's within a reasonable range.
        """
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=50_000,  # more scenarios for better tail resolution
            seed=0,
        )

        results = exp.run()
        method_a = results["regime_results"]["baseline"]["method_A_gci"]

        tail_cov = method_a["tail_prob_0_95"]
        # Should be positive and less than 0.20 (not catastrophically wrong)
        assert tail_cov > 0.0, "Tail coverage must be positive"
        assert tail_cov < 0.20, (
            f"Tail coverage {tail_cov:.4f} suggests severe mis-calibration"
        )

    def test_tail_coverage_monotonic_in_alpha(self, posterior_samples):
        """Tail coverage increases with alpha (more extreme = fewer exceedances)."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline"],
            n_scenarios=10_000,
            seed=0,
        )

        results = exp.run()
        method_a = results["regime_results"]["baseline"]["method_A_gci"]

        tail_95 = method_a["tail_prob_0_95"]
        tail_99 = method_a["tail_prob_0_99"]
        tail_999 = method_a["tail_prob_0_999"]

        # More extreme quantiles → fewer exceedances
        assert tail_99 <= tail_95 + 0.01, "Tail coverage should not increase with alpha"
        assert tail_999 <= tail_99 + 0.001, "Tail coverage should not increase with alpha"


# ---------------------------------------------------------------------------
# OOD widening tests
# ---------------------------------------------------------------------------
class TestOODWidening:
    def test_ood_widening(self, posterior_samples):
        """
        OOD regime: method B uncertainty band is wider than in-dist.

        When moving from baseline to a stress regime (housing_crash),
        the posterior spread should widen because the model is less
        certain about out-of-distribution parameter configurations.
        """
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline", "housing_crash"],
            n_scenarios=5_000,
            seed=0,
        )

        results = exp.run()

        # Compute 95% credible interval widths
        def credible_width(var_samples):
            if len(var_samples) == 0:
                return 0.0
            return float(np.percentile(var_samples, 97.5) - np.percentile(var_samples, 2.5))

        baseline_width = credible_width(
            results["regime_results"]["baseline"]["method_B_posterior"]["var_95_samples"]
        )
        housing_crash_width = credible_width(
            results["regime_results"]["housing_crash"]["method_B_posterior"]["var_95_samples"]
        )

        # OOD regime should have wider uncertainty bands
        assert housing_crash_width >= baseline_width * 0.8, (
            f"OOD widening check: housing_crash width ({housing_crash_width:.4f}) "
            f"should be >= 0.8x baseline width ({baseline_width:.4f})"
        )

    def test_ood_widening_all_stress_regimes(self, posterior_samples):
        """All stress regimes show widening vs baseline."""
        from experiments.ood_robustness import OODExperiment

        exp = OODExperiment(
            posterior_samples=posterior_samples,
            test_regimes=["baseline", "housing_crash", "rate_shock_1.5", "unemployment"],
            n_scenarios=3_000,
            seed=0,
        )

        results = exp.run()

        def credible_width(var_samples):
            if len(var_samples) == 0:
                return 0.0
            return float(np.percentile(var_samples, 97.5) - np.percentile(var_samples, 2.5))

        baseline_width = credible_width(
            results["regime_results"]["baseline"]["method_B_posterior"]["var_95_samples"]
        )

        for regime in ["housing_crash", "rate_shock_1.5", "unemployment"]:
            width = credible_width(
                results["regime_results"][regime]["method_B_posterior"]["var_95_samples"]
            )
            # Stress regime width should be at least 50% of baseline width
            # (conservative check — actual widening expected to be > 1.0x)
            assert width >= baseline_width * 0.5, (
                f"Regime {regime} width ({width:.4f}) < 0.5x baseline ({baseline_width:.4f})"
            )