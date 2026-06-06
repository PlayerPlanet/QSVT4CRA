"""
Gaussian factor copula for credit portfolio loss simulation.

Model
------
    Z       ~ N(0, 1)                 (systemic factor)
    ε_i     ~ N(0, 1)                 (idiosyncratic, independent)
    X_i     = b_i·Z + √(1-b_i²)·ε_i   (latent variable for loan i)
    default_i  iff  X_i < Φ⁻¹(p_i)
    U_i     = Φ(X_i)                  (uniform marginal)

θ layout (length 2K + 4)
-----------------------
    theta[0:K]           → factor_loadings  b_i
    theta[K:2*K]         → p_zeros          p_i (unconditional default probs)
    theta[2*K]           → tail_dep         (ignored for Gaussian)
    theta[2*K+1]         → rho              global correlation ρ
    theta[2*K+2]         → nu               (ignored; present for API compat)
    theta[2*K+3]         → spare            (unused)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


class GaussianFactorCopula:
    """
    One-factor Gaussian copula simulator.

    Generates uniform marginals and aggregated portfolio losses for a
    K-loan apartment loan portfolio using a single systemic factor Z.

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
        Draw n_samples from the Gaussian factor-copula model.

        Parameters
        ----------
        theta : np.ndarray, shape (2K + 4,)
            Parameter vector.  See class docstring for layout.
        n_samples : int, default 1000
            Number of scenarios.

        Returns
        -------
        U : np.ndarray, shape (n_samples, K), dtype float32
            Uniform marginals U[i, j] ~ Uniform(0, 1).
        losses : np.ndarray, shape (n_samples,), dtype float32
            Aggregated portfolio loss per scenario.
        """
        theta = self._validate_theta(theta)
        return self._sample_impl(theta, n_samples)

    def _sample_impl(
        self, theta: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Core sampling routine; factored out for subclass overrides."""
        K = self.K
        rng = self._rng

        # --- unpack theta ---
        b = theta[:K].astype(np.float64)          # factor loadings
        p = theta[K : 2 * K].astype(np.float64)   # default probs
        rho = float(theta[2 * K + 1])             # global correlation

        # --- systemic factor Z ~ N(0,1) ---
        Z = rng.standard_normal(size=n_samples).astype(np.float64)

        # --- idiosyncratic ε ~ N(0,1) shape (K, n_samples) ---
        eps = rng.standard_normal(size=(K, n_samples)).astype(np.float64)

        # --- latent variables X = b·Z + sqrt(1-b²)·ε ---
        # ensure |b| <= 1 (clip factor loadings for numerical stability)
        b_clipped = np.clip(b, -0.9999, 0.9999)
        sqrt_one_minus_b2 = np.sqrt(np.maximum(1.0 - b_clipped**2, 0.0))

        X = b_clipped[:, None] * Z[None, :] + sqrt_one_minus_b2[:, None] * eps

        # --- default flags: default_i iff X_i < Φ⁻¹(p_i) ---
        phi_inv_p = norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6))[:, None]  # (K, 1)
        defaults = (X < phi_inv_p).astype(np.float64)                # (K, n_samples)

        # --- uniform marginals U = Φ(X) ---
        U = norm.cdf(X).astype(np.float32)                            # (K, n_samples)
        U = U.T                                                       # (n_samples, K)

        # --- LGD per loan: fixed at 0.40 (midpoint of [0.20, 0.60]) ---
        lgd = np.full(K, 0.40, dtype=np.float32)

        # --- aggregated losses ---
        losses = (defaults.T * lgd[None, :]).sum(axis=1).astype(np.float32)

        return U, losses

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        """Validate theta shape and dtype."""
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
        """
        Convert uniform marginals to portfolio loss using explicit p_zeros.

        Parameters
        ----------
        U : np.ndarray, shape (n_samples, K)
            Uniform marginals.
        lgd : np.ndarray, shape (K,)
            LGD per loan.
        p_zeros : np.ndarray, shape (K,)
            Default thresholds (unconditional default probs).

        Returns
        -------
        losses : np.ndarray, shape (n_samples,)
        """
        U = np.asarray(U, dtype=np.float32)
        lgd = np.asarray(lgd, dtype=np.float32)
        p_zeros = np.asarray(p_zeros, dtype=np.float32)

        defaults = (U < p_zeros[None, :]).astype(np.float32)
        losses = (defaults * lgd[None, :]).sum(axis=1).astype(np.float32)
        return losses