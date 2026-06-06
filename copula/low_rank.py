"""
Low-rank factor copula for scalable credit portfolio loss simulation.

Model
------
    Z       ~ N(0, I_r)           (r-dimensional latent factor, r << K)
    ε_i     ~ N(0, 1)             (idiosyncratic, independent)
    X_i     = Σ_{s=1}^r A_{i,s}·Z_s + ε_i   (K-vector equation)
    default_i  iff  X_i < Φ⁻¹(p_i)
    U_i     = Φ(X_i)              (uniform marginal via Gaussian copula)

This is the **scalable** path for K up to 1000 (Phase 6) because the
factor dimension r is small (e.g., r=3), making the covariance matrix
K×K but rank-r, allowing efficient sampling via the factor structure.

θ layout (length K*r + K + 4)
----------------------------
    theta[0 : K*r]           → A matrix (K×r, row-major flatten)
    theta[K*r : K*r + K]     → p_zeros       (default probs)
    theta[K*r + K]           → tail_dep      (ignored)
    theta[K*r + K + 1]       → sigma_eps     (idiosyncratic std, default 1.0)
    theta[K*r + K + 2]       → spare_1
    theta[K*r + K + 3]       → spare_2
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


class LowRankFactorCopula:
    """
    Low-rank factor copula simulator.

    Uses a rank-r factor model to achieve O(r·n) sampling instead of
    O(K²·n) for full covariance.  Suitable for K = 100–1000 loans.

    Parameters
    ----------
    K : int
        Number of loans.
    r : int, default 3
        Rank of the factor model (r << K).
    seed : int, default 42
        Random seed.
    """

    def __init__(self, K: int, r: int = 3, seed: int = 42) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if r < 1 or r >= K:
            raise ValueError(f"r must satisfy 1 <= r < K, got r={r}")
        self.K = K
        self.r = r
        self._rng = np.random.default_rng(seed)

    def sample(
        self, theta: np.ndarray, n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw n_samples from the low-rank factor-copula model.

        Parameters
        ----------
        theta : np.ndarray, shape (K*r + K + 4,)
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
        r = self.r
        rng = self._rng

        # --- unpack theta ---
        A = theta[: K * r].astype(np.float64).reshape((K, r))   # (K, r)
        p = theta[K * r : K * r + K].astype(np.float64)        # (K,)
        # sigma_eps defaults to 1.0 if not explicitly set (theta slot is 0 by default)
        sigma_eps = float(theta[K * r + K + 1])
        if sigma_eps <= 0.0:
            sigma_eps = 1.0

        # --- sample latent factors Z ~ N(0, I_r) shape (n_samples, r) ---
        Z = rng.standard_normal(size=(n_samples, r)).astype(np.float64)  # (n_samples, r)

        # --- idiosyncratic shocks ε ~ N(0, sigma_eps²) shape (n_samples, K) ---
        eps = rng.normal(loc=0.0, scale=sigma_eps, size=(n_samples, K)).astype(np.float64)

        # --- latent variables X = Z·A.T + ε  (each row: n_samples → K) ---
        # Z is (n_samples, r), A is (K, r)
        # X = Z @ A.T + ε  →  (n_samples, K)
        X = Z @ A.T + eps

        # --- standardize to ensure uniform marginals ---
        # For U = Φ(X) to be uniform(0,1), X must be standard normal.
        # Standardize each column: X_std = (X - μ) / σ  (column-wise)
        X_mean = X.mean(axis=0)           # (K,)
        X_std = (X - X_mean) / (X.std(axis=0) + 1e-12)  # (n_samples, K)

        # --- default flags and uniform marginals ---
        # Defaults: standardized X < Φ⁻¹(p)  (probit threshold on standardized scale)
        phi_inv_p = stats.norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6))   # (K,)
        defaults = (X_std < phi_inv_p[None, :]).astype(np.float64)  # (n_samples, K)
        U = stats.norm.cdf(X_std).astype(np.float32)                 # (n_samples, K)

        # --- LGD: midpoint of [0.20, 0.60] ---
        lgd = np.full(K, 0.40, dtype=np.float32)
        losses = (defaults * lgd[None, :]).sum(axis=1).astype(np.float32)

        return U, losses

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        expected_len = self.K * self.r + self.K + 4
        if theta.shape != (expected_len,):
            raise ValueError(
                f"theta must have shape ({expected_len},) for K={self.K}, r={self.r}, "
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