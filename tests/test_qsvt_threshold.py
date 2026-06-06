"""
Tests for qsvt/threshold.py
"""
import numpy as np
import pytest

from qsvt.threshold import ThresholdFunction


class TestThresholdFunction:
    """Tests for ThresholdFunction class."""

    def test_polynomial_coefficients(self):
        """Test polynomial coefficient computation."""
        threshold = 0.5
        degree = 16
        tf = ThresholdFunction(threshold=threshold, degree=degree)

        coeffs = tf.polynomial_coefficients()

        assert isinstance(coeffs, np.ndarray)
        assert len(coeffs) == degree + 1
        assert coeffs.dtype == np.float32

    def test_polynomial_coefficients_length(self):
        """Test coefficients length matches degree."""
        for degree in [4, 8, 16, 32]:
            tf = ThresholdFunction(degree=degree)
            coeffs = tf.polynomial_coefficients()
            assert len(coeffs) == degree + 1

    def test_evaluate(self):
        """Test polynomial evaluation."""
        threshold = 0.5
        degree = 16
        tf = ThresholdFunction(threshold=threshold, degree=degree)

        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        values = tf.evaluate(x)

        assert isinstance(values, np.ndarray)
        assert len(values) == len(x)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_evaluate_bounds(self):
        """Test that evaluation is within [0, 1]."""
        tf = ThresholdFunction(threshold=0.5, degree=16)

        x = np.linspace(-1, 1, 100)
        values = tf.evaluate(x)

        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_qsvt_phases(self):
        """Test QSVT phase list generation."""
        tf = ThresholdFunction(threshold=0.5, degree=16)

        phases = tf.qsvt_phases()

        assert isinstance(phases, list)
        # pyqsp returns degree + 1 phases
        assert len(phases) == 17
        assert all(isinstance(p, (float, np.floating)) for p in phases)

    def test_qsvt_phases_high_degree(self):
        """Test QSVT phases for high degree (>= 256)."""
        tf = ThresholdFunction(threshold=0.5, degree=256)

        phases = tf.qsvt_phases()

        assert isinstance(phases, list)
        # pyqsp returns phases in a different format for high degree
        assert len(phases) > 0

    def test_threshold_edge_cases(self):
        """Test edge cases for threshold values."""
        # Valid thresholds should not raise
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            tf = ThresholdFunction(threshold=threshold, degree=8)
            assert tf.threshold == threshold

    def test_threshold_invalid(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError):
            ThresholdFunction(threshold=0.0, degree=8)

        with pytest.raises(ValueError):
            ThresholdFunction(threshold=1.0, degree=8)

    def test_degree_invalid(self):
        """Test that invalid degree raises error."""
        with pytest.raises(ValueError):
            ThresholdFunction(degree=0)

        with pytest.raises(ValueError):
            ThresholdFunction(degree=-1)

    def test_step_function_approximation(self):
        """Test that polynomial approximates step function."""
        threshold = 0.5
        degree = 32
        tf = ThresholdFunction(threshold=threshold, degree=degree)

        # Test points below and above threshold
        x_below = np.array([-1.0, -0.5, 0.0,0.3])
        x_above = np.array([0.7, 0.9, 1.0])

        values_below = tf.evaluate(x_below)
        values_above = tf.evaluate(x_above)

        # Values below threshold should be closer to 0
        assert np.mean(values_below) < np.mean(values_above)

    def test_different_degrees(self):
        """Test different polynomial degrees."""
        for degree in [4, 8, 16, 32, 64]:
            tf = ThresholdFunction(degree=degree)
            coeffs = tf.polynomial_coefficients()
            assert len(coeffs) == degree + 1
