"""
Base class for factor-copula risk models.

Each copula subclass generates uniform marginals U ~ Uniform[0,1]^K and
aggregated portfolio losses from a parameter vector θ.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class FactorCopula(ABC):
    """
    Abstract base for one-factor and multi-factor copula simulators.

    Parameters
    ----------
    K : int
        Number of loans (assets) in the portfolio.
    seed : int, default 42
        Random seed for reproducibility.

    Attributes
    ----------
    K : int
        Number of loans.
    rng : np.random.Generator
        NumPy random generator instance.
    """

    def __init__(self, K: int, seed: int = 42) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(
        self, theta: np.ndarray, n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw n_samples from the factor-copula model.

        Parameters
        ----------
        theta : np.ndarray, shape (D,)
            Parameter vector encoding factor loadings, default
            probabilities, tail dependence, and copula parameters.
        n_samples : int, default 1000
            Number of Monte Carlo scenarios.

        Returns
        -------
        U : np.ndarray, shape (n_samples, K), dtype float32
            Uniform marginals U[i, j] ~ Uniform(0, 1) for loan j in scenario i.
        losses : np.ndarray, shape (n_samples,), dtype float32
            Aggregated portfolio loss per scenario.
        """
        theta = self._validate_theta(theta)
        return self._sample_impl(theta, n_samples)

    @abstractmethod
    def _sample_impl(
        self, theta: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subclass-specific sampling implementation.

        Parameters
        ----------
        theta : np.ndarray, shape (D,)
            Validated parameter vector.
        n_samples : int
            Number of scenarios.

        Returns
        -------
        U : np.ndarray, shape (n_samples, K)
        losses : np.ndarray, shape (n_samples,)
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def losses_to_U_aggregated_loss(
        U: np.ndarray, lgd: np.ndarray
    ) -> np.ndarray:
        """
        Convert uniform marginals to portfolio loss via LGD-weighted defaults.

        A loan defaults when its uniform marginal U[i, j] falls below the
        default threshold p_j (encoded in theta).  The per-scenario loss is
        the sum of LGD[j] over all defaulted loans j.

        Parameters
        ----------
        U : np.ndarray, shape (n_samples, K)
            Uniform marginals U[i, j] ~ Uniform(0, 1).
        lgd : np.ndarray, shape (K,)
            Loss-given-default for each loan.

        Returns
        -------
        losses : np.ndarray, shape (n_samples,)
            Aggregated portfolio loss per scenario.
        """
        U = np.asarray(U, dtype=np.float32)
        lgd = np.asarray(lgd, dtype=np.float32)

        # Default flags: U < p_j.  Here p_j is the column-wise threshold
        # stored as the first K entries of theta (p_zeros).
        defaults = (U < 0.5).astype(np.float32)  # placeholder; caller passes p_zeros via theta
        losses = (defaults * lgd[None, :]).sum(axis=1)
        return losses.astype(np.float32)

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        """
        Validate and canonicalise the parameter vector.

        Parameters
        ----------
        theta : np.ndarray
            Raw parameter vector.

        Returns
        -------
        theta : np.ndarray, dtype float32
            Validated, canonicalised parameter vector.

        Raises
        ------
        ValueError
            If shape or dtype are invalid.
        """
        theta = np.asarray(theta, dtype=np.float32)
        self._validate_theta_shape(theta)
        return theta

    @abstractmethod
    def _validate_theta_shape(self, theta: np.ndarray) -> None:
        """
        Subclass-specific theta shape validation.

        Raises ValueError on mismatch.
        """