"""
Student-t factor copula for credit portfolio loss simulation with fat tails.

Model
------
    Z       ~ t_ν                     (systemic factor, ν degrees of freedom)
    ε_i     ~ N(0, 1)                 (idiosyncratic, independent)
    X_i     = b_i·Z + √(1-b_i²)·ε_i   (latent variable for loan i)
    default_i  iff  X_i < Φ⁻¹(p_i)     (probit link, same as Gaussian)
    U_i     = Φ(X_i)                  (uniform marginal via Gaussian copula)

Tail dependence
---------------
The t-copula exhibits symmetric tail dependence.  The upper/lower tail
dependence coefficient for a bivariate t-copula with correlation ρ and
ν degrees of freedom is:

    λ = 2⁻¹·T_{ν+2}(√(ν+1)·(√(1-ρ) - √(1+ρ))/2ρ)

where T_{ν+2} is the CDF of a standard t with ν+2 dof.

θ layout (length 2K + 4)
-----------------------
    theta[0:K]           → factor_loadings  b_i
    theta[K:2*K]         → p_zeros          p_i
    theta[2*K]           → tail_dep         (used to compute ν: smaller → fatter)
    theta[2*K+1]         → rho              global correlation ρ
    theta[2*K+2]         → nu              degrees of freedom ν
    theta[2*K+3]         → spare
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


class StudentTFactorCopula:
    """
    One-factor Student-t copula simulator with symmetric tail dependence.

    Parameters
    ----------
    K : int
        Number of loans.
    seed : int, default 42
        Random seed.
    """

    def __init__(self, K: int, seed: int = 42) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
        self._rng = np.random.default_rng(seed)

    def sample(
        self, theta: np.ndarray, n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw n_samples from the Student-t factor-copula model.

        Parameters
        ----------
        theta : np.ndarray, shape (2K + 4,)
            Parameter vector.  See class docstring for layout.
        n_samples : int, default 1000
            Number of scenarios.

        Returns
        -------
        U : np.ndarray, shape (n_samples, K), dtype float32
            Uniform marginals.
        losses : np.ndarray, shape (n_samples,), dtype float32
            Aggregated portfolio loss per scenario.
        """
        theta = self._validate_theta(theta)
        return self._sample_impl(theta, n_samples)

    def _sample_impl(
        self, theta: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = self.K
        rng = self._rng

        # --- unpack theta ---
        b = theta[:K].astype(np.float64)
        p = theta[K : 2 * K].astype(np.float64)
        tail_dep = float(theta[2 * K])
        rho = float(theta[2 * K + 1])
        nu = float(theta[2 * K + 2])

        # --- map tail_dep to nu (inverse: smaller nu → fatter tails) ---
        # tail_dep ∈ [0, 1]; 0 → Gaussian (nu=100), 1 → heavy tail (nu=2)
        nu_effective = max(2.0, 100.0 - 98.0 * tail_dep)

        # --- sample systemic factor Z ~ t_nu ---
        # Z = W / sqrt(V/nu)  with  W~N(0,1), V~χ²(nu)
        W = rng.standard_normal(size=n_samples).astype(np.float64)
        V = rng.chisquare(df=nu_effective, size=n_samples).astype(np.float64)
        Z = W / np.sqrt(V / nu_effective)                                # (n_samples,)

        # --- idiosyncratic ε ~ N(0,1) shape (K, n_samples) ---
        eps = rng.standard_normal(size=(K, n_samples)).astype(np.float64)

        # --- latent variables ---
        b_clipped = np.clip(b, -0.9999, 0.9999)
        sqrt_one_minus_b2 = np.sqrt(np.maximum(1.0 - b_clipped**2, 0.0))

        X = b_clipped[:, None] * Z[None, :] + sqrt_one_minus_b2[:, None] * eps

        # --- default flags and uniform marginals ---
        phi_inv_p = stats.norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6))[:, None]
        defaults = (X < phi_inv_p).astype(np.float64)
        U = stats.norm.cdf(X).astype(np.float32)
        U = U.T                                                      # (n_samples, K)

        # --- LGD: midpoint of [0.20, 0.60] ---
        lgd = np.full(K, 0.40, dtype=np.float32)
        losses = (defaults.T * lgd[None, :]).sum(axis=1).astype(np.float32)

        return U, losses

    @staticmethod
    def tail_dependence_coefficient(rho: float, nu: float) -> float:
        """
        Compute the upper/lower tail dependence coefficient for a bivariate
        t-copula with correlation rho and nu degrees of freedom.

        The formula (Demarta & McNeil 2005) is::

            λ = 2 * T_{ν+2}(-√((ν+1)*(1-ρ)/(1+ρ))) / √(1-ρ)

        where T_{ν+2} is the standard t CDF with ν+2 dof.

        For ρ = 0 (independence), the t-copula has no tail dependence (λ = 0).
        For ρ → 1, λ → 1 (full tail dependence).

        Parameters
        ----------
        rho : float
            Correlation coefficient in [-1, 1].
        nu : float
            Degrees of freedom (> 2).

        Returns
        -------
        lambda_ : float
            Tail dependence coefficient in [0, 1].
        """
        if nu <= 2.0:
            return 1.0
        if abs(rho) >= 1.0:
            return 1.0 if abs(rho) == 1.0 else 0.0

        # Independence case: ρ ≈ 0 → λ ≈ 0
        if abs(rho) < 1e-8:
            return 0.0

        # Full formula: λ = 2 * (1+ρ)/√(1-ρ) * T_{ν+2}(-√((ν+1)*(1-ρ)/(1+ρ)))
        sqrt_ratio = np.sqrt((1.0 - rho) / (1.0 + rho + 1e-12))
        t_arg = -np.sqrt(nu + 1.0) * sqrt_ratio
        lam = (
            2.0
            * (1.0 + rho)
            / np.sqrt(1.0 - rho + 1e-12)
            * stats.t.cdf(t_arg, df=nu + 2.0)
        )
        return float(np.clip(lam, 0.0, 1.0))

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        expected_len = 2 * self.K + 4
        if theta.shape != (expected_len,):
            raise ValueError(
                f"theta must have shape ({expected_len},) for K={self.K}, "
                f"got shape {theta.shape}"
            )
        return theta

    @staticmethod
    def losses_to_U_aggregated_loss(
        U: np.ndarray, lgd: np.ndarray, p_zeros: np.ndarray
    ) -> np.ndarray:
        """Convert uniform marginals to portfolio loss using explicit p_zeros."""
        U = np.asarray(U, dtype=np.float32)
        lgd = np.asarray(lgd, dtype=np.float32)
        p_zeros = np.asarray(p_zeros, dtype=np.float32)

        defaults = (U < p_zeros[None, :]).astype(np.float32)
        losses = (defaults * lgd[None, :]).sum(axis=1).astype(np.float32)
        return losses