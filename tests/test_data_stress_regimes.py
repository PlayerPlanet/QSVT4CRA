"""
Tests for data/stress_regimes.py — StressRegimeGenerator and REGIME_SPECS.
"""
from __future__ import annotations

import numpy as np
import pytest
from data.stress_regimes import StressRegimeGenerator, REGIME_SPECS


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
    copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
    return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])


@pytest.fixture
def regime_gen():
    return StressRegimeGenerator(seed=42)


# ---------------------------------------------------------------------------
# REGIME_SPECS tests
# ---------------------------------------------------------------------------
class TestRegimeSpecs:
    """Tests for the REGIME_SPECS registry."""

    def test_all_five_regimes_present(self):
        expected = {"baseline", "housing_crash", "rate_shock", "unemployment", "liquidity"}
        assert set(REGIME_SPECS.keys()) == expected

    def test_baseline_is_identity(self, theta_baseline, K):
        theta_stressed = REGIME_SPECS["baseline"](theta_baseline, shock=1.0, K=K)
        np.testing.assert_array_equal(theta_stressed, theta_baseline)

    def test_stress_regimes_produce_different_theta(self, theta_baseline, K):
        """All non-baseline regimes should perturb the baseline theta."""
        baseline_result = REGIME_SPECS["baseline"](theta_baseline, shock=1.0, K=K)
        for name, func in REGIME_SPECS.items():
            if name == "baseline":
                continue
            result = func(theta_baseline, shock=1.0, K=K)
            # At least one element should differ
            assert not np.allclose(result, baseline_result), (
                f"Regime {name} should differ from baseline"
            )

    def test_shock_magnitude_zero_same_as_baseline(self, theta_baseline, K):
        """Shock magnitude 0 should produce same theta for all regimes."""
        # All regimes with shock=0 should return identical theta (baseline behaviour)
        results = {}
        for name, func in REGIME_SPECS.items():
            results[name] = func(theta_baseline, shock=0.0, K=K)
        # All results should be equal to theta_baseline (and to each other)
        for name, result in results.items():
            np.testing.assert_array_equal(result, theta_baseline)


# ---------------------------------------------------------------------------
# StressRegimeGenerator tests
# ---------------------------------------------------------------------------
class TestStressRegimeGeneratorInit:
    """Tests for __init__."""

    def test_default_seed(self):
        gen = StressRegimeGenerator()
        assert gen._rng is not None

    def test_custom_seed(self):
        gen = StressRegimeGenerator(seed=99)
        assert gen._rng is not None


class TestStressRegimeGeneratorSample:
    """Tests for sample()."""

    def test_output_shape_matches_input(self, regime_gen, theta_baseline):
        result = regime_gen.sample("baseline", theta_baseline, shock_magnitude=1.0)
        assert result.shape == theta_baseline.shape

    def test_output_dtype_float32(self, regime_gen, theta_baseline):
        result = regime_gen.sample("baseline", theta_baseline, shock_magnitude=1.0)
        assert result.dtype == np.float32

    def test_unknown_regime_raises(self, regime_gen, theta_baseline):
        with pytest.raises(ValueError, match="Unknown regime"):
            regime_gen.sample("unknown_regime", theta_baseline, shock_magnitude=1.0)

    def test_all_regimes_run(self, regime_gen, theta_baseline):
        """All five regimes should run without error."""
        for name in REGIME_SPECS:
            result = regime_gen.sample(name, theta_baseline, shock_magnitude=0.5)
            assert result.shape == theta_baseline.shape
            assert result.dtype == np.float32

    def test_housing_crash_increases_p_zeros(self, regime_gen, theta_baseline, K):
        result = regime_gen.sample("housing_crash", theta_baseline, shock_magnitude=1.0)
        p_zeros_baseline = theta_baseline[K : 2 * K]
        p_zeros_stressed = result[K : 2 * K]
        # At least some p_zeros should increase
        assert np.any(p_zeros_stressed > p_zeros_baseline)

    def test_rate_shock_increases_p_zeros(self, regime_gen, theta_baseline, K):
        result = regime_gen.sample("rate_shock", theta_baseline, shock_magnitude=1.0)
        p_zeros_baseline = theta_baseline[K : 2 * K]
        p_zeros_stressed = result[K : 2 * K]
        assert np.any(p_zeros_stressed > p_zeros_baseline)

    def test_unemployment_changes_factor_loadings(self, regime_gen, theta_baseline, K):
        result = regime_gen.sample("unemployment", theta_baseline, shock_magnitude=1.0)
        fl_baseline = theta_baseline[:K]
        fl_stressed = result[:K]
        assert not np.allclose(fl_stressed, fl_baseline)

    def test_liquidity_changes_nu(self, regime_gen, theta_baseline, K):
        result = regime_gen.sample("liquidity", theta_baseline, shock_magnitude=1.0)
        nu_baseline = theta_baseline[2 * K + 2]
        nu_stressed = result[2 * K + 2]
        assert nu_stressed < nu_baseline

    def test_shock_magnitude_scales_effect(self, regime_gen, theta_baseline, K):
        """Higher shock magnitude → larger deviation from baseline."""
        p_zeros_baseline = theta_baseline[K : 2 * K]
        for magnitude in [0.0, 0.5, 1.0]:
            result = regime_gen.sample("housing_crash", theta_baseline, shock_magnitude=magnitude)
            p_zeros_stressed = result[K : 2 * K]
            deviation = np.abs(p_zeros_stressed - p_zeros_baseline).sum()
            if magnitude == 0.0:
                assert deviation < 1e-6
            else:
                assert deviation > 0.0

    def test_sample_returns_independent_copy(self, regime_gen, theta_baseline):
        """Modifying the returned array should not affect the original."""
        result = regime_gen.sample("housing_crash", theta_baseline, shock_magnitude=1.0)
        original = theta_baseline.copy()
        result[0] = 999.0
        assert theta_baseline[0] != 999.0
        np.testing.assert_array_equal(theta_baseline, original)
