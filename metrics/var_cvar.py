"""
Classical risk metrics: Value-at-Risk (VaR) and Conditional VaR (CVaR).

VaR_α  = the α-quantile of the loss distribution (e.g., VaR_0.95 = 95th percentile)
CVaR_α = the expected loss given that loss exceeds VaR_α (Expected Shortfall)

These are the benchmark metrics against which QSVT approximations are validated.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def loss_cdf(losses: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """
    Empirical cumulative distribution function (CDF) of portfolio losses.

    The empirical CDF at point x is the fraction of loss samples ≤ x.

    Parameters
    ----------
    losses : np.ndarray, shape (N,)
        Monte Carlo loss samples.
    x_grid : np.ndarray, shape (M,)
        Grid points at which to evaluate the CDF.

    Returns
    -------
    cdf : np.ndarray, shape (M,)
        Empirical CDF values in [0, 1].

    Examples
    --------
    >>> losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> x_grid = np.array([0.0, 2.5, 5.0])
    >>> loss_cdf(losses, x_grid)
    array([0. , 0.4, 1. ])
    """
    losses = np.asarray(losses, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64)

    # Broadcast: losses (N,) vs x_grid (M,) -> (M, N)
    cdf = np.sum(losses[None, :] <= x_grid[:, None], axis=1) / losses.shape[0]
    return cdf.astype(np.float32)


def var_at(losses: np.ndarray, alpha: float) -> float:
    """
    Value-at-Risk at confidence level α.

    VaR_α is the α-quantile of the loss distribution, i.e., the loss value
    such that P(loss ≤ VaR_α) = α. Also known as the quantile function.

    For a sorted loss array L_{(1)} ≤ L_{(2)} ≤ ... ≤ L_{(N)}, VaR_α = L_{(⌈αN⌉)}.

    Parameters
    ----------
    losses : np.ndarray, shape (N,)
        Monte Carlo loss samples.
    alpha : float
        Confidence level in (0, 1).  Common values: 0.95, 0.99, 0.999.

    Returns
    -------
    var : float
        VaR at level α (same units as losses).

    Raises
    ------
    ValueError
        If alpha is not in (0, 1) or losses is empty.

    Notes
    -----
    - VaR_0.95 < VaR_0.99 < VaR_0.999 (monotonically non-decreasing in α)
    - VaR is a quantile, not an expectation.
    - For discrete empirical distributions, the quantile definition uses
 linear interpolation (numpy default) to avoid discontinuities.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        raise ValueError("losses must not be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    return float(np.quantile(losses, alpha))


def cvar_at(losses: np.ndarray, alpha: float) -> float:
    """
    Conditional Value-at-Risk (CVaR) / Expected Shortfall at confidence level α.

    CVaR_α = E[loss | loss ≥ VaR_α] — the expected loss given that the loss
    exceeds the α-quantile.  Also known as Expected Shortfall.

    Computed as the mean of all loss samples strictly above the α-quantile.

    Parameters
    ----------
    losses : np.ndarray, shape (N,)
        Monte Carlo loss samples.
    alpha : float
        Confidence level in (0, 1).  Common values: 0.95, 0.99, 0.999.

    Returns
    -------
    cvar : float
        CVaR at level α (same units as losses).

    Raises
    ------
    ValueError
        If alpha is not in (0, 1) or losses is empty.

    Notes
    -----
    - CVaR_α ≥ VaR_α for all α (by definition of conditional expectation)
    - CVaR is more sensitive to tail risk than VaR alone.
    - For continuous distributions, CVaR is the mean excess function.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        raise ValueError("losses must not be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    var = np.quantile(losses, alpha)
    # Include samples strictly above VaR (tail)
    tail_samples = losses[losses > var]
    if tail_samples.size == 0:
        # All remaining samples at or below VaR (degenerate case)
        return var
    return float(tail_samples.mean())


def var_cvar(
    losses: np.ndarray,
    alphas: list = None,
) -> dict:
    """
    Compute VaR and CVaR at multiple confidence levels in one call.

    Parameters
    ----------
    losses : np.ndarray, shape (N,)
        Monte Carlo loss samples.
    alphas : list of float, optional
        Confidence levels. Defaults to [0.95, 0.99, 0.999].

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - ``var_0.95``, ``var_0.99``, ``var_0.999`` : VaR at each level
        - ``cvar_0.95``, ``cvar_0.99``, ``cvar_0.999`` : CVaR at each level
        - ``tail_prob_0.95``, ``tail_prob_0.99``, ``tail_prob_0.999`` :
          P(loss > VaR_α) ≈ 1 - α

    Notes
    -----
    - ``var_0.95`` means VaR at 95% confidence (5% tail)
    - ``cvar_0.95`` is the expected loss in that5% tail
    - The tail probability is computed empirically from the sample.
    """
    if alphas is None:
        alphas = [0.95, 0.99, 0.999]

    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        raise ValueError("losses must not be empty")

    metrics = {}
    for alpha in alphas:
        var = var_at(losses, alpha)
        cvar = cvar_at(losses, alpha)
        tail_prob = float(np.mean(losses > var))

        # Sanitize key: replace "." with "_" for dict key compatibility
        key_base = str(alpha).replace(".", "_")
        metrics[f"var_{key_base}"] = var
        metrics[f"cvar_{key_base}"] = cvar
        metrics[f"tail_prob_{key_base}"] = tail_prob

    return metrics
