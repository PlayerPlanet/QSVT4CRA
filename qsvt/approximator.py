"""
QSVTApproximator — QSVT polynomial phase sequence computation.

Wraps pyqsp.angle_sequence.QuantumSignalProcessingPhases for low-degree
polynomials and provides a hand-rolled Chebyshev fallback for high-degree
(>=256) polynomials.

Architecture: D4 (docs/architecture.md §4)
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


class QSVTApproximator:
    """
    QSVT phase sequence approximator using pyqsp.

    Wraps pyqsp.angle_sequence.QuantumSignalProcessingPhases to compute
    phase sequences for QSVT polynomial application.

    Parameters
    ----------
    prior : dict, optional
        Prior parameters for the approximation (unused in current impl).
    """

    def __init__(self, prior: Optional[dict] = None) -> None:
        self.prior = prior or {}

    def approximate_threshold(
        self,
        threshold: float,
        degree: int,
        target_loss: float = 0.5,
        max_loss: float = 1.0,
    ) -> List[float]:
        """
        Compute QSVT phase sequence for threshold function approximation.

        Parameters
        ----------
        threshold : float
            Threshold value in [0, 1].
        degree : int
            Polynomial degree.
        target_loss : float, default 0.5
            Target loss value.
        max_loss : float, default 1.0
            Maximum loss for normalization.

        Returns
        -------
        phases : list of float
            Phase sequence for QSVT application.
        """
        from qsvt.threshold import ThresholdFunction

        tf = ThresholdFunction(
            threshold=threshold,
            degree=degree,
            target_loss=target_loss,
            max_loss=max_loss,
        )
        return tf.qsvt_phases()


class ChebyshevApproximator:
    """
    Hand-rolled Chebyshev phase sequence for high-degree QSVT.

    Used as fallback when pyqsp fails at degree >= 256.

    Parameters
    ----------
    degree : int
        Polynomial degree.
    """

    def __init__(self, degree: int) -> None:
        self.degree = degree

    def compute_phases(
        self,
        poly_coeffs: np.ndarray,
        threshold: float = 0.5,
    ) -> List[float]:
        """
        Compute Chebyshev phase sequence from polynomial coefficients.

        Parameters
        ----------
        poly_coeffs : np.ndarray
            Polynomial coefficients in Chebyshev basis.
        threshold : float, default 0.5
            Threshold value.

        Returns
        -------
        phases : list of float
            Phase sequence for QSVT.
        """
        degree = self.degree
        coeffs = np.asarray(poly_coeffs, dtype=np.float64)

        # Normalize coefficients
        norm = np.sqrt(np.sum(coeffs**2))
        if norm > 0:
            coeffs = coeffs / norm

        phases = []
        c0 = coeffs[0] if len(coeffs) > 0 else 1.0
        if c0 == 0:
            c0 = 1.0

        for n in range(degree):
            c_n = coeffs[n] if n < len(coeffs) else 0.0
            ratio = c_n / c0
            ratio = np.clip(ratio, -1.0, 1.0)

            if abs(ratio) > 1e-10:
                phase = np.arccos(ratio)
            else:
                phase = 0.0

            phases.append(float(phase))

        return phases


def approximate_threshold(
    threshold: float,
    degree: int,
    target_loss: float = 0.5,
    max_loss: float = 1.0,
) -> List[float]:
    """
    Compute QSVT phase sequence for threshold function approximation.

    This is the main entry point for phase sequence computation.

    Parameters
    ----------
    threshold : float
        Threshold value in [0, 1].
    degree : int
        Polynomial degree.
    target_loss : float, default 0.5
        Target loss value.
    max_loss : float, default 1.0
        Maximum loss for normalization.

    Returns
    -------
    phases : list of float
        Phase sequence for QSVT application.

    Notes
    -----
    - For degree <= 256, uses pyqsp for accurate phase computation
    - For degree > 256, uses hand-rolled Chebyshev fallback
    - pyqsp 0.2.0 degree limits are respected (max ~1024 in practice)
    """
    if degree <= 256:
        approximator = QSVTApproximator()
        return approximator.approximate_threshold(
            threshold=threshold,
            degree=degree,
            target_loss=target_loss,
            max_loss=max_loss,
        )
    else:
        # Hand-rolled fallback for high degree
        from qsvt.threshold import ThresholdFunction

        tf = ThresholdFunction(
            threshold=threshold,
            degree=degree,
            target_loss=target_loss,
            max_loss=max_loss,
        )
        coeffs = tf.polynomial_coefficients()
        cheby_approx = ChebyshevApproximator(degree=degree)
        return cheby_approx.compute_phases(coeffs, threshold=threshold)
