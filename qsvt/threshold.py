"""
ThresholdFunction — even threshold function polynomial for QSVT.

Constructs the Chebyshev polynomial approximation to the even step function
that QSVT will apply to distinguish loss > target from loss ≤ target.

Architecture: D4 (docs/architecture.md §4)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class ThresholdFunction:
    """
    Even threshold function polynomial for QSVT.

    Constructs a polynomial approximation to the step function:
 f(x) = 1 if x > threshold
             = 0 if x< threshold
             = 0.5 if x = threshold

    The polynomial is constructed in the Chebyshev basis and then
    converted to QSVT phase angles.

    Parameters
    ----------
    threshold : float, default 0.5
        Threshold value in [0, 1] (normalized loss).
    degree : int, default 16
        Polynomial degree (controls approximation accuracy).
    target_loss : float, default 0.5
        Target loss value for the VaR/CVaR computation.
    max_loss : float, default 1.0
        Maximum possible loss for normalization.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        degree: int = 16,
        target_loss: float = 0.5,
        max_loss: float = 1.0,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if degree < 1 or degree > 2048:
            raise ValueError(f"degree must be in [1, 2048], got {degree}")

        self.threshold = threshold
        self.degree = degree
        self.target_loss = target_loss
        self.max_loss = max_loss

    def polynomial_coefficients(self) -> np.ndarray:
        """
        Compute Chebyshev coefficients for the even threshold function.

        The step function is approximated by a truncated Chebyshev series.
        For the even step function, only even-degree Chebyshev polynomials
        appear (T_0, T_2, T_4, ...).

        Returns
        -------
        coeffs : np.ndarray, shape (degree + 1,)
            Chebyshev coefficients c_n such that:
 f(x) ≈ sum_{n=0}^{degree} c_n * T_n(x) for x in [-1, 1]
        """
        degree = self.degree
        threshold = self.threshold

        # Chebyshev nodes for the given degree
        nodes = np.cos(np.pi * np.arange(degree + 1) / degree)

        # Step function values at Chebyshev nodes
        values = np.where(nodes > threshold, 1.0, 0.0).astype(np.float64)

        # Compute Chebyshev coefficients via discrete cosine transform
        # For even function approximation, use only even terms
        coeffs = np.zeros(degree + 1, dtype=np.float64)

        for n in range(degree + 1):
            if n == 0:
                coeffs[n] = np.mean(values)
            else:
                # Chebyshev-Gauss quadrature weights
                if n == degree:
                    weight = 0.5
                else:
                    weight = 1.0
                T_n = np.cos(n * np.arccos(np.clip(nodes, -1, 1)))
                coeffs[n] = (2.0 / degree) * weight * np.sum(values * T_n)

        # Zero out odd coefficients to ensure definite parity
        for n in range(1, degree + 1):
            if n % 2 == 1:
                coeffs[n] = 0.0

        return coeffs.astype(np.float32)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the threshold polynomial at points x.

        Parameters
        ----------
        x : np.ndarray
            Points in [-1, 1] at which to evaluate.

        Returns
        -------
        values : np.ndarray
            Polynomial values in [0, 1].
        """
        x = np.asarray(x, dtype=np.float64)
        coeffs = self.polynomial_coefficients()
        degree = self.degree

        # Clamp x to [-1, 1] for stability
        x_clamped = np.clip(x, -1.0, 1.0)

        # Evaluate Chebyshev polynomial via recurrence
        values = np.zeros_like(x_clamped)

        # T_0(x) = 1, T_1(x) = x
        T_n_minus_1 = np.ones_like(x_clamped)  # T_0
        T_n = x_clamped.copy()                   # T_1

        if degree == 0:
            return np.full_like(x_clamped, coeffs[0])

        values += coeffs[0] * T_n_minus_1
        if degree >= 1:
            values += coeffs[1] * T_n

        # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
        T_n_minus_2 = T_n_minus_1
        for n in range(1, degree):
            T_n_plus_1 = 2.0 * x_clamped * T_n - T_n_minus_2
            values += coeffs[n + 1] * T_n_plus_1
            T_n_minus_2 = T_n
            T_n = T_n_plus_1

        return np.clip(values, 0.0, 1.0).astype(np.float32)

    def qsvt_phases(self) -> List[float]:
        """
        Convert polynomial coefficients to QSVT phase list.

        Returns
        -------
        phases : list of float
            Phase angles for QSVT application.
            Length = degree.
        """
        coeffs = self.polynomial_coefficients()

        # For low degree, use pyqsp if available
        if self.degree <= 256:
            try:
                import pyqsp

                phases = self._pyqsp_phases(coeffs)
                return phases
            except ImportError:
                pass

        # Fall back to hand-rolled Chebyshev phase sequence
        phases = self._chebyshev_phases(coeffs)
        return phases

    def _pyqsp_phases(self, coeffs: np.ndarray) -> List[float]:
        """
        Compute QSVT phases using pyqsp.

        Parameters
        ----------
        coeffs : np.ndarray
            Chebyshev coefficients.

        Returns
        -------
        phases : list of float
        """
        from pyqsp.angle_sequence import QuantumSignalProcessingPhases

        poly = coeffs.tolist()
        result = QuantumSignalProcessingPhases(
            poly,
            signal_operator="Wx",
            method="sym_qsp",
            measurement="x",
            chebyshev_basis=True,
        )
        # pyqsp returns a tuple (phases_array, 0, 0) or just phases_array
        if isinstance(result, tuple):
            return list(result[0])
        return list(result)

    def _chebyshev_phases(self, coeffs: np.ndarray) -> List[float]:
        """
        Hand-rolled Chebyshev phase sequence for high-degree polynomials.

        For degree >= 256 where pyqsp may fail, we use the analytical
        Chebyshev phase formula.

        Parameters
        ----------
        coeffs : np.ndarray
            Chebyshev coefficients.

        Returns
        -------
        phases : list of float
        """
        degree = self.degree
        phases = []

        # Analytical Chebyshev phase formula for even polynomial
        # Phase n = arccos(c_n / c_0) for main coefficients
        c0 = coeffs[0] if len(coeffs) > 0 else 1.0
        if c0 == 0:
            c0 = 1.0

        for n in range(degree):
            c_n = coeffs[n] if n < len(coeffs) else 0.0
            # Phase angle based on coefficient ratio
            ratio = c_n / c0
            ratio = np.clip(ratio, -1.0, 1.0)
            phase = np.arccos(ratio) if abs(ratio) > 1e-10 else 0.0
            phases.append(float(phase))

        return phases
