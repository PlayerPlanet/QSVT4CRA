"""
Tests for StudentTFactorCopula.

Coverage targets:
- shape and dtype (same as Gaussian)
- marginal U ~ Uniform(0,1)
- losses aggregation
- reproducibility
- tail dependence coefficient computation
- low-dof fat-tail check: at ν=2, tail_dep > 0.1
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from copula.student_t import StudentTFactorCopula


class TestStudentTFactorCopula:
    """Unit tests for StudentTFactorCopula."""

    def test_sample_shape_and_dtype(self):
        """sample() returns correct shapes and float32 dtype."""
        K = 10
        cop = StudentTFactorCopula(K=K, seed=42)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.3
        theta[K : 2 * K] = 0.02
        theta[2 * K] = 0.5     # tail_dep → nu ≈ 51
        theta[2 * K + 1] = 0.5  # rho
        theta[2 * K + 2] = 4.0  # nu (dof)

        U, losses = cop.sample(theta, n_samples=500)

        assert U.shape == (500, K)
        assert losses.shape == (500,)
        assert U.dtype == np.float32
        assert losses.dtype == np.float32

    def test_marginals_uniform(self):
        """Generated U marginals are consistent with Uniform(0,1)."""
        K = 10
        cop = StudentTFactorCopula(K=K, seed=99)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.4
        theta[K : 2 * K] = 0.05
        theta[2 * K] = 0.5   # tail_dep
        theta[2 * K + 1] = 0.3
        theta[2 * K + 2] = 6.0

        U, _ = cop.sample(theta, n_samples=2000)

        for j in range(K):
            d_stat, p_val = stats.kstest(U[:, j], "uniform", args=(0.0, 1.0))
            assert p_val > 0.01, f"Marginal {j} fails KS test (p={p_val:.4f})"

    def test_reproducibility(self):
        """Fixed seed + same theta → identical outputs."""
        K = 8
        cop1 = StudentTFactorCopula(K=K, seed=777)
        cop2 = StudentTFactorCopula(K=K, seed=777)

        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.5
        theta[K : 2 * K] = 0.03
        theta[2 * K] = 0.5
        theta[2 * K + 1] = 0.4
        theta[2 * K + 2] = 5.0

        U1, losses1 = cop1.sample(theta, n_samples=200)
        U2, losses2 = cop2.sample(theta, n_samples=200)

        np.testing.assert_array_equal(U1, U2)
        np.testing.assert_array_equal(losses1, losses2)

    def test_theta_validation_wrong_length(self):
        """Wrong-length theta raises ValueError."""
        cop = StudentTFactorCopula(K=5, seed=0)
        bad_theta = np.zeros(3, dtype=np.float32)

        with pytest.raises(ValueError, match="shape"):
            cop.sample(bad_theta, n_samples=10)

    def test_tail_dependence_coefficient(self):
        """tail_dependence_coefficient returns values in [0, 1]."""
        cop = StudentTFactorCopula.__new__(StudentTFactorCopula)

        # Gaussian limit (nu → ∞): λ → 0
        lam_gaussian = StudentTFactorCopula.tail_dependence_coefficient(rho=0.5, nu=100.0)
        assert lam_gaussian < 0.1

        # Heavy tail (low ν): λ > 0
        lam_heavy = StudentTFactorCopula.tail_dependence_coefficient(rho=0.5, nu=3.0)
        assert 0.0 < lam_heavy <= 1.0

        # Very heavy tail (ν=2): λ should be substantial
        lam_vheavy = StudentTFactorCopula.tail_dependence_coefficient(rho=0.7, nu=2.0)
        assert lam_vheavy > 0.1, f"Expected tail_dep > 0.1 for ν=2, got {lam_vheavy:.4f}"

    def test_tail_dependence_coefficient_boundary(self):
        """Boundary cases for tail_dependence_coefficient."""
        cop = StudentTFactorCopula.__new__(StudentTFactorCopula)

        # rho = 0 → λ = 0
        lam = StudentTFactorCopula.tail_dependence_coefficient(rho=0.0, nu=5.0)
        assert lam < 1e-6

        # nu <= 2 → λ = 1 (full tail dependence)
        lam = StudentTFactorCopula.tail_dependence_coefficient(rho=0.5, nu=2.0)
        assert lam >= 0.99

    def test_low_dof_fat_tails_vs_gaussian(self):
        """Student-t with low ν produces heavier tails than Gaussian."""
        K = 5
        n = 5000

        theta_gauss = np.zeros(2 * K + 4, dtype=np.float32)
        theta_gauss[:K] = 0.7   # strong factor loading
        theta_gauss[K : 2 * K] = 0.05
        theta_gauss[2 * K] = 0.0   # tail_dep = 0 → Gaussian (nu=100)
        theta_gauss[2 * K + 1] = 0.7
        theta_gauss[2 * K + 2] = 100.0

        theta_t = np.zeros(2 * K + 4, dtype=np.float32)
        theta_t[:K] = 0.7
        theta_t[K : 2 * K] = 0.05
        theta_t[2 * K] = 1.0   # tail_dep = 1 → ν ≈ 2 (fat tails)
        theta_t[2 * K + 1] = 0.7
        theta_t[2 * K + 2] = 2.0

        cop_gauss = StudentTFactorCopula(K=K, seed=42)
        cop_t = StudentTFactorCopula(K=K, seed=42)

        _, losses_gauss = cop_gauss.sample(theta_gauss, n_samples=n)
        _, losses_t = cop_t.sample(theta_t, n_samples=n)

        # t-copula should have higher 99th percentile loss (fatter right tail)
        q99_gauss = np.percentile(losses_gauss, 99)
        q99_t = np.percentile(losses_t, 99)
        assert q99_t >= q99_gauss * 0.9, "t-copula tail should be at least as heavy"

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

        losses = StudentTFactorCopula.losses_to_U_aggregated_loss(U, lgd, p_zeros)

        expected = np.array([0.30, 0.40, 0.50, 0.00], dtype=np.float32)
        np.testing.assert_array_almost_equal(losses, expected)