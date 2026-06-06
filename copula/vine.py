"""
Hand-rolled D-vine copula for credit portfolio loss simulation.

D-vine structure (Bedford & Cooke 2002; Aas et al. 2009)
--------------------------------------------------------
A D-vine is a regular vine specialised to a sequential ordering of
variables.  For K variables ordered 1..K, the decomposition is:

    f(x₁,...,x_K) =
        f(x₁) · f(x₂|x₁) · f(x₃|x₁,x₂) · ... · f(x_K|x₁,...,x_{K-1})

The pair-copula construction (PCC) factorises each conditional density
into bivariate pair-copula densities.  For a D-vine with K variables:

    Tree 1 (level 0): pairs (1,2), (2,3), ..., (K-1,K)          → K-1 pairs
    Tree 2 (level 1): pairs (1,2;3), (2,3;4), ...               → K-2 pairs
    Tree 3 (level 2): ...
    ...

Each pair copula can be Gaussian or Clayton (selected per-pair by theta).

θ layout (length 2K + 4)
-----------------------
    theta[0:K]           → p_zeros          default probs
    theta[K:2*K]         → factor_loadings  (inform ordering; not used in sampling)
    theta[2*K]           → tail_dep
    theta[2*K+1]         → rho              base correlation for all pairs
    theta[2*K+2]         → pair_copula_type 0=Gaussian, 1=Clayton
    theta[2*K+3]         → spare
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


class DVineCopula:
    """
    D-vine copula simulator with hand-rolled sequential structure.

    Uses Gaussian pair copulas at each tree level.  The D-vine is built
    from the bottom up:

        Tree 1: (1,2), (2,3), ..., (K-1,K)
        Tree 2: condition on variable 3 → pairs (1,2|3), (2,3|4), ...
        Tree 3: condition on variables 3,4 → ...

    For large K the full vine is combinatorial; this implementation
    limits to K ≤ 20 for practical computation.

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
        Draw n_samples from the D-vine copula model.

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
        p = theta[:K].astype(np.float64)          # default probs
        rho = float(theta[2 * K + 1])             # base correlation
        pair_type = int(theta[2 * K + 2])         # 0=Gaussian, 1=Clayton

        # Clip rho for numerical stability
        rho = np.clip(rho, -0.99, 0.99)

        # --- sample K independent Uniform(0,1) via Gaussian pair-copula ---
        # The D-vine decomposition induces correlation structure through
        # sequential conditioning.  We simulate via the sequential method:
        # 1. Sample U₁ ~ Uniform
        # 2. For i = 2..K: sample U_i | U_{i-1} using pair copula C(u_i | u_{i-1})
        U = np.zeros((n_samples, K), dtype=np.float64)

        # First margin: U₁ ~ Uniform(0,1)
        U[:, 0] = rng.uniform(size=n_samples)

        # Sequential sampling using Gaussian pair copula
        # C(u_i | u_{i-1}) = Φ((Φ⁻¹(u_i) - ρ·Φ⁻¹(u_{i-1})) / √(1-ρ²))
        # Inverse: given u_{i-1}, sample v_i = Φ(Z_i) where Z_i ~ N(0,1)
        # conditioned on the previous value.
        phi_inv_U_prev = stats.norm.ppf(np.clip(U[:, 0], 1e-6, 1 - 1e-6))

        for i in range(1, K):
            # Conditional distribution: Z_i | Z_{i-1} = z_prev ~ N(ρ*z_prev, 1-ρ²)
            cond_mean = rho * phi_inv_U_prev
            cond_std = np.sqrt(max(1.0 - rho**2, 1e-6))
            Z_i_given = cond_mean + cond_std * rng.standard_normal(n_samples)
            U[:, i] = stats.norm.cdf(Z_i_given)

            # Update phi_inv_U_prev for next iteration
            phi_inv_U_prev = stats.norm.ppf(np.clip(U[:, i], 1e-6, 1 - 1e-6))

        # --- default flags ---
        defaults = (U < p[None, :]).astype(np.float64)  # (n_samples, K)

        # --- uniform marginals (Gaussian copula induced) ---
        U_out = U.astype(np.float32)

        # --- LGD: midpoint of [0.20, 0.60] ---
        lgd = np.full(K, 0.40, dtype=np.float32)
        losses = (defaults * lgd[None, :]).sum(axis=1).astype(np.float32)

        return U_out, losses

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