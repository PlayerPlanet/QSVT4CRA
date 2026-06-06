"""
conftest.py - Test configuration for QSVT4CRA Phase 1 tests.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def K():
    return 10


@pytest.fixture
def theta_baseline(K):
    """Baseline theta for K=10. Explicitly returns a fresh copy."""
    rng = np.random.default_rng(0)
    factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
    p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
    tail_dep = np.array(0.0, dtype=np.float32)
    copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
    result = np.concatenate(
        [factor_loadings, p_zeros, [tail_dep], copula_params]
    ).copy()
    return result


@pytest.fixture
def regime_gen():
    return pytest.importorskip("data.stress_regimes").StressRegimeGenerator(seed=42)
