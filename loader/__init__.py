"""
Loader package — quantum circuit loaders for posterior factor copula.

Modules
-------
posterior_factor_copula : PosteriorFactorCopulaLoader (replaces MultivariateGCI)
amplitude_loader : AmplitudeLoader (efficient amplitude encoding)
"""
from __future__ import annotations

from loader.posterior_factor_copula import PosteriorFactorCopulaLoader
from loader.amplitude_loader import AmplitudeLoader

__all__ = [
    "PosteriorFactorCopulaLoader",
    "AmplitudeLoader",
]
