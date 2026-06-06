"""
Parametric stress regime definitions and generator.

Applies regime-dependent shocks to a baseline θ vector, producing
perturbed parameters for stress testing the factor-copula model.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Regime specifications
# ---------------------------------------------------------------------------
def _baseline(theta: np.ndarray, shock: float, K: int) -> np.ndarray:
    """Identity (no shock)."""
    return theta.copy()


def _housing_crash(theta: np.ndarray, shock: float, K: int) -> np.ndarray:
    """
    Housing price crash: increase default probabilities and factor loadings.

    - p_zeros scaled by (1 + 0.3*shock) to (1 + 1.0*shock)
    - factor_loadings amplified (housing-sensitive loans bear the brunt)
    - tail_dep increased (more co-movement in downturn)
    """
    if shock <= 0.0:
        return theta.copy()
    theta_star = theta.copy()
    p_zeros = theta_star[K : 2 * K]
    factor_loadings = theta_star[:K]

    # Scale p_zeros upward: 30–100% increase
    scale = 1.0 + (0.3 + 0.7 * np.clip(shock, 0.0, 1.0))
    theta_star[K : 2 * K] = np.clip(p_zeros * scale, 0.001, 0.95)

    # Amplify negative loadings (most housing-sensitive)
    adjustment = np.where(factor_loadings < 0, 0.25 * shock, 0.10 * shock)
    theta_star[:K] = factor_loadings * (1.0 + adjustment)

    # Increase tail dependence
    theta_star[2 * K] = min(theta_star[2 * K] * (1.0 + 0.5 * shock), 5.0)

    # Copula: increase correlation
    theta_star[2 * K + 1] = min(theta_star[2 * K + 1] + 0.20 * shock, 0.95)

    return theta_star


def _rate_shock(theta: np.ndarray, shock: float, K: int) -> np.ndarray:
    """
    Interest rate shock: raise default probabilities and tighten credit.

    - p_zeros scaled by (1 + 0.5*shock) to (1 + 3.0*shock)
    - correlations increased (credit market stress)
    """
    if shock <= 0.0:
        return theta.copy()
    theta_star = theta.copy()
    p_zeros = theta_star[K : 2 * K]

    # Higher scaling than housing crash
    scale = 1.0 + (0.5 + 2.5 * np.clip(shock, 0.0, 1.0))
    theta_star[K : 2 * K] = np.clip(p_zeros * scale, 0.001, 0.99)

    # Increase copula correlation
    theta_star[2 * K + 1] = min(theta_star[2 * K + 1] + 0.15 * shock, 0.95)

    # Slightly increase tail dependence
    theta_star[2 * K] = min(theta_star[2 * K] + 0.3 * shock, 5.0)

    return theta_star


def _unemployment(theta: np.ndarray, shock: float, K: int) -> np.ndarray:
    """
    Unemployment spike: regional p_zeros doubled, factor loadings shifted.

    - Loans in Oulu/Turku (more industrial) see larger default increases
    - factor_loadings reweighted toward regional unemployment sensitivity
    """
    if shock <= 0.0:
        return theta.copy()
    theta_star = theta.copy()
    p_zeros = theta_star[K : 2 * K]
    factor_loadings = theta_star[:K]

    # Double p_zeros for bottom half of factor loadings (more exposed regions)
    sorted_idx = np.argsort(factor_loadings)
    n_stressed = max(1, int(K * 0.5))
    stress_idx = sorted_idx[:n_stressed]

    p_zeros_star = p_zeros.copy()
    p_zeros_star[stress_idx] = np.clip(
        p_zeros[stress_idx] * (1.0 + 1.0 * shock), 0.001, 0.99
    )
    theta_star[K : 2 * K] = p_zeros_star

    # Shift factor loadings toward unemployment-sensitive loans
    theta_star[:K] = factor_loadings * (1.0 - 0.15 * shock)

    # Copula correlation increases
    theta_star[2 * K + 1] = min(theta_star[2 * K + 1] + 0.25 * shock, 0.95)

    return theta_star


def _liquidity(theta: np.ndarray, shock: float, K: int) -> np.ndarray:
    """
    Liquidity crisis: LGDs raised, recovery delayed.

    - LGD parameter is implicit in factor loadings via the tail_dep slot.
    We represent LGD stress by increasing tail_dep and lowering nu (dof).
    - Correlations increase.
    """
    if shock <= 0.0:
        return theta.copy()
    theta_star = theta.copy()

    # Increase tail dependence (proxy for higher LGD / recovery risk)
    theta_star[2 * K] = min(theta_star[2 * K] * (1.0 + 0.4 * shock), 5.0)

    # Decrease degrees of freedom (fatter tails = higher effective LGD)
    nu = theta_star[2 * K + 2]
    theta_star[2 * K + 2] = max(nu - 2.0 * shock, 3.0)

    # Increase correlation
    theta_star[2 * K + 1] = min(theta_star[2 * K + 1] + 0.15 * shock, 0.95)

    return theta_star


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
REGIME_SPECS: Dict[str, Callable[[np.ndarray, float, int], np.ndarray]] = {
    "baseline": _baseline,
    "housing_crash": _housing_crash,
    "rate_shock": _rate_shock,
    "unemployment": _unemployment,
    "liquidity": _liquidity,
}


# ---------------------------------------------------------------------------
# StressRegimeGenerator
# ---------------------------------------------------------------------------
class StressRegimeGenerator:
    """
    Generate stressed parameter vectors from a baseline θ.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        regime_name: str,
        theta_baseline: np.ndarray,
        shock_magnitude: float = 1.0,
    ) -> np.ndarray:
        """
        Apply a named stress regime to theta_baseline.

        Parameters
        ----------
        regime_name : str
            One of: ``"baseline"``, ``"housing_crash"``, ``"rate_shock"``,
            ``"unemployment"``, ``"liquidity"``.
        theta_baseline : np.ndarray, shape (D,)
            Baseline parameter vector.
        shock_magnitude : float, default 1.0
            Shock intensity in [0, 1].0 = baseline, 1 = full stress.

        Returns
        -------
        np.ndarray, shape (D,)
            Perturbed parameter vector.
        """
        if regime_name not in REGIME_SPECS:
            raise ValueError(
                f"Unknown regime {regime_name!r}. "
                f"Valid: {list(REGIME_SPECS.keys())}"
            )
        theta_baseline = np.asarray(theta_baseline, dtype=np.float32)
        K = (theta_baseline.shape[0] - 4) // 2
        pert = REGIME_SPECS[regime_name](theta_baseline, shock_magnitude, K)
        return pert.astype(np.float32)
