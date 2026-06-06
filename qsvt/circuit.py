"""
QSVTRiskCircuit — full QSVT risk circuit composition.

Composes PosteriorFactorCopulaLoader + threshold amplitude loading + QSVT
into a complete circuit for VaR/CVaR computation.

Compatible with get_expected_probability_circuit(uncertainity_model=loader, ...).

Architecture: D4 (docs/architecture.md §4)
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit import QuantumRegister

from loader.posterior_factor_copula import PosteriorFactorCopulaLoader
from qsvt.approximator import approximate_threshold


class QSVTRiskCircuit(QuantumCircuit):
    """
    Full QSVT risk circuit for posterior-propagated VaR/CVaR computation.

    Composes:
    1. PosteriorFactorCopulaLoader — state preparation U(θ)
    2. Threshold amplitude loading — maps loss to angle
    3. QSVT application — polynomial approximation

    Parameters
    ----------
    loader : PosteriorFactorCopulaLoader
        The posterior factor copula loader circuit.
    target_loss : float
        Target loss threshold for VaR/CVaR.
    degree : int
        QSVT polynomial degree.
    threshold : float, default 0.5
        Threshold value for the step function.
    name : str, optional
        Circuit name. Default "QSVTrisk".

    Attributes
    ----------
    num_qubits : int
        Total number of qubits in the circuit.
    """

    def __init__(
        self,
        loader: PosteriorFactorCopulaLoader,
        target_loss: float,
        degree: int,
        threshold: float = 0.5,
        name: str = "QSVTrisk",
    ) -> None:
        self._loader = loader
        self._target_loss = target_loss
        self._degree = degree
        self._threshold = threshold

        # Get dimensions from loader
        K = loader.K
        lgd = loader.lgd

        # Build registers
        state_q = QuantumRegister(K, name="state")
        target_q = QuantumRegister(1, name="target")
        aux_q = QuantumRegister(1, name="aux")

        super().__init__(state_q, target_q, aux_q, name=name)

        # Build the circuit
        self._build_circuit(K, lgd, target_loss, degree, threshold)

    def _build_circuit(
        self,
        K: int,
        lgd: np.ndarray,
        target_loss: float,
        degree: int,
        threshold: float,
    ) -> None:
        """Build the full QSVT risk circuit."""
        # Step 1: State preparation via loader
        loader_gate = self._loader.to_gate()
        self.append(loader_gate, self.qubits[:K + 1])

        # Step 2: Compute QSVT phases
        phases = approximate_threshold(
            threshold=threshold,
            degree=degree,
            target_loss=target_loss,
            max_loss=float(np.sum(lgd)),
        )

        # Step 3: Build QSVT circuit using Code/QSVT.QSVT
        from Code.QSVT import QSVT

        # Create the objective circuit (AmplitudeLoadingVar)
        objective_circuit = self._build_objective_circuit(K, lgd, target_loss, threshold)

        # QSVT requires subspace qubits and control settings
        qsvt_circuit = QSVT(
            objective_circuit,
            subspace_qubits1=[K],  # target qubit index
            subspace_qubits2=[K],
            phases=phases,
            adjust_conventions=True,
            ctrl_zero_qubits1=[0],
            ctrl_zero_qubits2=[0],
            name="QSVT",
        )

        # Append QSVT circuit
        self.append(qsvt_circuit.to_gate(), self.qubits)

    def _build_objective_circuit(
        self,
        K: int,
        lgd: np.ndarray,
        target_loss: float,
        threshold: float,
    ) -> QuantumCircuit:
        """
        Build the objective circuit for loss comparison.

        This is the AmplitudeLoadingVar circuit that maps loss values
        to angles for the QSVT.

        Parameters
        ----------
        K : int
            Number of state qubits.
        lgd : np.ndarray
            LGD array.
        target_loss : float
            Target loss threshold.
        threshold : float
            Threshold value.

        Returns
        -------
        QuantumCircuit
            The objective circuit.
        """
        from Code.AmplitudeLoading import AmplitudeLoadingVar

        offsets = lgd
        maximum = float(np.sum(offsets))
        target_loss_scaled = target_loss / maximum
        arc_threshold = np.arcsin(threshold)
        maximum_angle = np.pi / 2
        minimum_angle = 0

        # Compute unitary gap
        if target_loss_scaled > (arc_threshold - minimum_angle) / (maximum_angle - minimum_angle):
            minimum_range = minimum_angle
            unitary_gap = (arc_threshold - minimum_range) / target_loss_scaled
        else:
            maximum_range = maximum_angle
            unitary_gap = (maximum_range - arc_threshold) / (1 - target_loss_scaled)

        # Transform losses to angles
        def transform_loss_to_angle(loss: float) -> float:
            return unitary_gap * (loss / maximum - target_loss_scaled) + arc_threshold

        scaled_offsets = [
            transform_loss_to_angle(off) - transform_loss_to_angle(0)
            for off in offsets
        ]

        # Build the amplitude loading circuit
        objective = AmplitudeLoadingVar(
            K,
            scaled_offsets,
            starting_offset=transform_loss_to_angle(0),
            name="objective",
        )

        return objective

    @property
    def num_qubits(self) -> int:
        """Total number of qubits."""
        return len(self.qubits)

    @property
    def K(self) -> int:
        """Number of state qubits."""
        return self._loader.K

    def _decompose(self) -> QuantumCircuit:
        """
        Return decomposed circuit.

        Returns
        -------
        QuantumCircuit
            Circuit decomposed to basis gates.
        """
        return self.decompose()
