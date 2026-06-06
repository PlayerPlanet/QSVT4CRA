"""
Tests for qsvt/approximator.py
"""
import numpy as np
import pytest

from qsvt.approximator import (
    QSVTApproximator,
    ChebyshevApproximator,
    approximate_threshold,
)


class TestQSVTApproximator:
    """Tests for QSVTApproximator class."""

    def test_approximate_threshold(self):
        """Test threshold approximation."""
        approximator = QSVTApproximator()

        phases = approximator.approximate_threshold(
            threshold=0.5,
            degree=16,
            target_loss=0.5,
            max_loss=1.0,
        )

        assert isinstance(phases, list)
        # pyqsp returns degree + 1 phases
        assert len(phases) == 17
        assert all(isinstance(p, (float, np.floating)) for p in phases)

    def test_pyqsp_wrapper(self):
        """Test that pyqsp wrapper produces valid phases."""
        approximator = QSVTApproximator()

        phases = approximator.approximate_threshold(
            threshold=0.5,
            degree=16,
        )

        # Phases should be in reasonable range
        for phase in phases:
            assert -np.pi <= phase <= np.pi


class TestChebyshevApproximator:
    """Tests for ChebyshevApproximator class."""

    def test_compute_phases(self):
        """Test Chebyshev phase computation."""
        degree = 32
        approximator = ChebyshevApproximator(degree=degree)

        coeffs = np.random.randn(degree + 1).astype(np.float64)
        phases = approximator.compute_phases(coeffs, threshold=0.5)

        assert isinstance(phases, list)
        assert len(phases) == degree

    def test_high_degree_fallback(self):
        """Test high degree fallback."""
        degree = 256
        approximator = ChebyshevApproximator(degree=degree)

        coeffs = np.random.randn(degree + 1).astype(np.float64)
        phases = approximator.compute_phases(coeffs, threshold=0.5)

        assert isinstance(phases, list)
        assert len(phases) == degree


class TestApproximateThreshold:
    """Tests for the approximate_threshold function."""

    def test_low_degree(self):
        """Test low degree approximation."""
        phases = approximate_threshold(
            threshold=0.5,
            degree=16,
        )

        assert isinstance(phases, list)
        # pyqsp returns degree + 1 phases
        assert len(phases) == 17

    def test_high_degree(self):
        """Test high degree approximation (>= 256)."""
        phases = approximate_threshold(
            threshold=0.5,
            degree=256,
        )

        assert isinstance(phases, list)
        # pyqsp returns degree + 1 phases
        assert len(phases) == 257

    def test_different_thresholds(self):
        """Test different threshold values."""
        for threshold in [0.3, 0.5, 0.7]:
            phases = approximate_threshold(
                threshold=threshold,
                degree=16,
            )
            # pyqsp returns degree + 1 phases
            assert len(phases) == 17

    def test_phase_range(self):
        """Test that phases are in valid range."""
        phases = approximate_threshold(
            threshold=0.5,
            degree=32,
        )

        for phase in phases:
            assert -np.pi <= phase <= np.pi
