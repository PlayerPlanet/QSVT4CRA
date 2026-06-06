"""
Data package for QSVT4CRA research run.
Provides synthetic portfolio generation, stress regime perturbations,
and real data loader hooks.
"""
from .synthetic import SyntheticPortfolioGenerator, PortfolioDataset
from .stress_regimes import StressRegimeGenerator, REGIME_SPECS
from .loader import RealDataLoader

__all__ = [
    "SyntheticPortfolioGenerator",
    "PortfolioDataset",
    "StressRegimeGenerator",
    "REGIME_SPECS",
    "RealDataLoader",
]
