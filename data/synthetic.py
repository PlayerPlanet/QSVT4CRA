"""
Synthetic portfolio generator with known ground-truth θ.

Produces Finnish-style apartment loan portfolios using a one-factor Gaussian
or Student-t copula, enabling end-to-end validation of SBI posterior recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Region definitions (Finnish apartment markets)
# ---------------------------------------------------------------------------
REGIONS = ("Helsinki", "Tampere", "Turku", "Oulu", "Other")

# ---------------------------------------------------------------------------
# PortfolioDataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioDataset:
    """
    Container for a single portfolio simulation draw.

    Attributes
    ----------
    losses : float32[n_scenarios]
        Aggregated portfolio losses (EUR, unscaled) per scenario.
    observations : float32[n_scenarios, 10]
        Summary statistics extracted from each scenario's loss vector.
        Columns: n_defaults, mean_lgd, std_lgd, helsinki_rate, tampere_rate,
        turku_rate, oulu_rate, factor_z_mean, factor_z_std, var95.
    theta : float32[D]
        Ground-truth parameter vector used to generate this dataset.
    """

    losses: np.ndarray
    observations: np.ndarray
    theta: np.ndarray


# ---------------------------------------------------------------------------
# SyntheticPortfolioGenerator
# ---------------------------------------------------------------------------
class SyntheticPortfolioGenerator:
    """
    Finnish apartment loan portfolio generator with known ground-truth θ.

    Uses a one-factor copula (Gaussian or Student-t) to model joint default
    dependence across K loans. The ground-truth parameter vector θ encodes
    factor loadings, marginal default probabilities, tail-dependence strength,
    and copula parameters.

    θ layout (length D = 2*K + 4):
        theta[0:K]               → factor_loadings  (how each loan loads on Z)
        theta[K:2*K]             → p_zeros          (unconditional default probs)
        theta[2*K]               → tail_dep         (0=Gaussian, >0=Student-t)
        theta[2*K+1:2*K+4]       → copula_params    (rho, nu_dof, rotation, spare)

    Parameters
    ----------
    K : int, default 10
        Number of loans in the portfolio.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, K: int = 10, seed: Optional[int] = None) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
        self._rng = np.random.default_rng(seed)

        # Loan-level attributes (fixed across draws, derived from theta via init)
        self._lgd: np.ndarray = np.zeros(K, dtype=np.float32)
        self._principal: np.ndarray = np.zeros(K, dtype=np.float32)
        self._region_idx: np.ndarray = np.zeros(K, dtype=np.intp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(self, theta: np.ndarray, n_scenarios: int) -> PortfolioDataset:
        """
        Draw n_scenarios from the factor-copula model with parameter vector θ.

        Parameters
        ----------
        theta : np.ndarray, shape (D,)
            Parameter vector. See class docstring for layout.
        n_scenarios : int
            Number of Monte Carlo scenarios to draw.

        Returns
        -------
        PortfolioDataset
            Named tuple containing ``losses``, ``observations``, and ``theta``.
        """
        theta = self._validate_theta(theta)
        self._derive_loan_attributes(theta)

        losses = self._simulate_losses(theta, n_scenarios)
        observations = self.losses_to_observation(losses)

        return PortfolioDataset(
            losses=losses,
            observations=observations,
            theta=theta.astype(np.float32),
        )

    def losses_to_observation(self, losses: np.ndarray) -> np.ndarray:
        """
        Transform a loss array into a single 10-dimensional observation vector.

        The observation vector aggregates statistics across all n_scenarios
        into one summary per theta (used by the forward simulator).

        Columns
        -------
        0  n_defaults       : mean number of defaulted loans across scenarios
        1  mean_lgd        : mean LGD across defaulted loans (0 if none)
        2  std_lgd         : std LGD across defaulted loans (0 if <2 defaults)
        3  helsinki_rate : mean fraction of Helsinki-region defaults
        4  tampere_rate    : mean fraction of Tampere-region defaults
        5  turku_rate      : mean fraction of Turku-region defaults
        6  oulu_rate       : mean fraction of Oulu-region defaults
        7  factor_z_mean   : mean of factor Z proxy across scenarios
        8  factor_z_std    : std of factor Z proxy across scenarios
        9  var95           : 95th percentile of portfolio loss

        Parameters
        ----------
        losses : np.ndarray, shape (n_scenarios,)
            Aggregated portfolio losses per scenario.

        Returns
        -------
        np.ndarray, shape (10,)
            Single observation vector (float32) aggregating all scenarios.
        """
        losses = np.asarray(losses, dtype=np.float32)
        n = losses.shape[0]

        # Region indices and masks
        region_idx = self._region_idx
        masks = {r: region_idx == i for r, i in zip(REGIONS, range(len(REGIONS)))}

        # Mean LGD per loan (used for default count proxy)
        mean_lgd_per_loan = self._lgd.mean()

        # n_defaults proxy: inferred from loss magnitude relative to mean LGD
        n_defaults_est = np.clip(
            (losses / (mean_lgd_per_loan + 1e-8)).astype(np.float32),
            0.0,
            float(self.K),
        )

        # Regional default rates (mean across scenarios)
        regional_rates = np.zeros(4, dtype=np.float32)
        for col_idx, region in enumerate(("Helsinki", "Tampere", "Turku", "Oulu")):
            mask = masks[region]
            n_region = mask.sum()
            if n_region > 0:
                region_exposure = self._principal[mask].sum()
                rate = (losses / (region_exposure + 1e-8)).clip(0.0, 1.0)
                regional_rates[col_idx] = rate.mean()

        # Factor Z proxy moments
        expected_loss = mean_lgd_per_loan * self._lgd.sum()
        factor_z = (losses - expected_loss) / (expected_loss + 1e-8)
        factor_z_mean = factor_z.mean()
        factor_z_std = factor_z.std()

        # VaR at 95%
        var95 = np.percentile(losses, 95).astype(np.float32)

        # Build single observation vector
        obs = np.array([
            n_defaults_est.mean(),   # col 0: mean n_defaults
            0.0,                     # col 1: mean_lgd (placeholder)
            0.0,                     # col 2: std_lgd (placeholder)
            regional_rates[0],       # col 3: helsinki_rate
            regional_rates[1],       # col 4: tampere_rate
            regional_rates[2],       # col 5: turku_rate
            regional_rates[3],       # col 6: oulu_rate
            factor_z_mean,           # col 7: factor_z_mean
            factor_z_std,            # col 8: factor_z_std
            var95,                   # col 9: var95
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        """Ensure theta is float32 with the correct shape."""
        theta = np.asarray(theta, dtype=np.float32)
        expected_len = 2 * self.K + 4
        if theta.shape != (expected_len,):
            raise ValueError(
                f"theta must have shape ({expected_len},) for K={self.K}, "
                f"got shape {theta.shape}"
            )
        return theta

    def _derive_loan_attributes(self, theta: np.ndarray) -> None:
        """
        Initialise per-loan LGD, principal, and region from theta.

        Uses a deterministic mapping from factor_loadings to create
        heterogeneous but reproducible loan characteristics.
        """
        factor_loadings = theta[: self.K]
        p_zeros = theta[self.K : 2 * self.K]

        rng = self._rng

        # LGD: uniform in [0.20, 0.60], informed by factor loading sign
        lgd_base = rng.uniform(0.20, 0.60, size=self.K).astype(np.float32)
        # Negative loadings → slightly higher LGD (stress-sensitive loans)
        lgd_adjustment = np.where(factor_loadings < 0, 0.05, 0.0)
        self._lgd = np.clip(lgd_base + lgd_adjustment, 0.10, 0.80).astype(
            np.float32
        )

        # Principal: €50k–€500k, log-uniform for realistic skew
        log_principal = rng.uniform(
            np.log(50e3), np.log(500e3), size=self.K
        )
        self._principal = np.exp(log_principal).astype(np.float32)

        # Region: assign based on factor loading ranking
        sorted_indices = np.argsort(factor_loadings)[::-1]
        region_assignments = np.array(
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 4][: self.K], dtype=np.intp
        )
        self._region_idx = region_assignments[sorted_indices]

    def _simulate_losses(
        self, theta: np.ndarray, n_scenarios: int
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation for the one-factor copula.

        Draws Z ~ N(0,1), computes conditional default probabilities via
        the probit link, draws Bernoulli defaults, and aggregates LGD-weighted
        losses.
        """
        from scipy.stats import norm

        K = self.K
        factor_loadings = theta[:K].astype(np.float64)
        p_zeros = theta[K : 2 * K].astype(np.float64)
        tail_dep = float(theta[2 * K])
        rho, nu, rotation = theta[2 * K + 1 : 2 * K + 4].astype(np.float64)

        rng = self._rng
        lgd = self._lgd.astype(np.float64)
        principal = self._principal.astype(np.float64)

        # --- Systemic factor Z ---
        if tail_dep > 1e-6 and nu < 100:  # Student-t copula
            # Sample Z via t-distribution: Z = W / sqrt(V/nu) where W~N(0,1), V~chi2(nu)
            W = rng.standard_normal(size=n_scenarios).astype(np.float64)
            V = rng.chisquare(df=nu, size=n_scenarios).astype(np.float64)
            Z = W / np.sqrt(V / nu)
        else:  # Gaussian copula
            Z = rng.standard_normal(size=n_scenarios).astype(np.float64)

        # --- Conditional default probabilities via probit link ---
        # p_i|Z = Phi((Phi^{-1}(p_i) - sqrt(rho)*beta_i*Z) / sqrt(1 - rho*beta_i^2))
        sqrt_rho = np.sqrt(np.clip(rho, 0.0, 0.9999))
        beta_sq = factor_loadings**2
        denom = np.sqrt(1.0 - rho * beta_sq + 1e-12)

        # phi_inv_p = norm.ppf(p_zeros) for each loan
        phi_inv_p = norm.ppf(np.clip(p_zeros, 1e-6, 1.0 - 1e-6))

        # numerator per loan: phi_inv_p - sqrt(rho)*beta*Z
        numerator = phi_inv_p[:, None] - sqrt_rho * factor_loadings[:, None] * Z[None, :]
        cond_p = norm.cdf(numerator / denom[:, None])

        # --- Bernoulli default draws ---
        U = rng.uniform(size=(K, n_scenarios)).astype(np.float64)
        defaults = (U < cond_p).astype(np.float64)  # shape (K, n_scenarios)

        # --- Per-loan losses: LGD * principal * default ---
        # loss_i = lgd_i * principal_i * default_i (loss in EUR)
        loan_losses = lgd[:, None] * principal[:, None] * defaults  # (K, n_scenarios)

        # --- Aggregate portfolio loss ---
        total_losses = loan_losses.sum(axis=0)  # (n_scenarios,)

        return total_losses.astype(np.float32)
