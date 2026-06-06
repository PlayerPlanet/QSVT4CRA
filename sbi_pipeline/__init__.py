"""SBI Posterior Estimation Package for QSVT4CRA.

This package provides Neural Posterior Estimation (NPE), Neural Likelihood
Estimation (NLE), and Flow Matching training pipelines for constructing
posterior distributions over factor-copula parameters given apartment loan
portfolio loss data.

Recommended SOTA stack: sbi 0.22.0 + NPE + TMNRE
Reference: arXiv:1905.07488 (APT), arXiv:2210.02747 (Flow Matching)
"""

from .posterior import (
    NPETrainingPipeline,
    NLETrainingPipeline,
    FlowMatchingTrainingPipeline,
    SBIPosterior,
)
from .utils import SBITrainingConfig, get_prior_from_bounds

__all__ = [
    "NPETrainingPipeline",
    "NLETrainingPipeline",
    "FlowMatchingTrainingPipeline",
    "SBIPosterior",
    "SBITrainingConfig",
    "get_prior_from_bounds",
]
