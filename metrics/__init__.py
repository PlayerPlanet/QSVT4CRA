"""
metrics — Classical risk metrics for QSVT4CRA ground truth.

Public API
----------
var_cvar      : compute VaR and CVaR at multiple confidence levels
var_at        : α-quantile of a loss distribution
cvar_at       : expected loss given loss exceeds VaR_α
loss_cdf      : empirical CDF of losses on a grid
GroundTruthMC : Monte Carlo ground truth engine over posterior samples
"""
from __future__ import annotations

from .ground_truth import GroundTruthMC
from .var_cvar import cvar_at, loss_cdf, var_at, var_cvar

__all__ = [
    "var_cvar",
    "var_at",
    "cvar_at",
    "loss_cdf",
    "GroundTruthMC",
]
