"""
Copula package — factor-copula risk models for QSVT4CRA.

Copula families
---------------
GaussianFactorCopula  : one-factor Gaussian copula (Gaussian tail, no tail dep)
StudentTFactorCopula   : one-factor Student-t copula (symmetric fat tails)
DVineCopula           : hand-rolled D-vine pair-copula construction
LowRankFactorCopula   : low-rank (r << K) factor model for large portfolios

Unified interface
-----------------
All copulas expose:

    sample(theta, n_samples) -> (U: float32[n_samples, K], losses: float32[n_samples])

where:
    U[i, j] ~ Uniform(0, 1) is the j-th loan's uniform marginal in scenario i
    losses[i] = sum_j (default_ij * lgd_j)  aggregated portfolio loss

Theta layout (for Gaussian, Student-t, D-vine)
--------------------------------------------
    theta[0:K]           → factor_loadings  b_i
    theta[K:2*K]         → p_zeros          p_i (unconditional default probs)
    theta[2*K]           → tail_dep
    theta[2*K+1]         → rho              global correlation ρ
    theta[2*K+2]         → nu              degrees of freedom (Student-t only)
    theta[2*K+3]         → spare

Theta layout (for LowRankFactorCopula)
--------------------------------------
    theta[0:K*r]         → A matrix (K×r, row-major flatten)
    theta[K*r:K*r+K]    → p_zeros
    theta[K*r+K]        → tail_dep
    theta[K*r+K+1]       → sigma_eps
    theta[K*r+K+2..]    → spare
"""
from __future__ import annotations

from .gaussian import GaussianFactorCopula
from .student_t import StudentTFactorCopula
from .vine import DVineCopula
from .low_rank import LowRankFactorCopula

__all__ = [
    "GaussianFactorCopula",
    "StudentTFactorCopula",
    "DVineCopula",
    "LowRankFactorCopula",
]