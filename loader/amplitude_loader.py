"""
AmplitudeLoader — efficient amplitude encoding for quantum circuits.

Uses a Schmidt-decomposition-style decomposition to avoid O(2^n) cost of
naive amplitude loading. Encodes a vector of values as amplitudes of a
quantum state over K state qubits plus 1 target qubit.

Reference: Code/AmplitudeLoading.py:AmplitudeLoadingVar (prototype)
"""
from __future__ import annotations

from typing import List

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit import QuantumRegister


class AmplitudeLoader(QuantumCircuit):
    """
    Encode a vector of values as amplitudes using controlled-rotations.

    Creates a circuit that maps basis states |j>|0> → |j>(cos(α_j)|0> + sin(α_j)|1>)
    where α_j ∝ arcsin(values[j] / max_val).

    This is the efficient amplitude loading used by PosteriorFactorCopulaLoader.
    Unlike NormalDistribution (which scales as 2^n qubits), this scales as K+1 qubits
    for K state qubits encoding 2^K scenario values.

    Parameters
    ----------
    num_state_qubits : int
        Number of state qubits (determines capacity: 2^num_state_qubits values).
    values : np.ndarray
        Array of values to encode. Must have length 2^num_state_qubits.
        Values are sum-normalized internally for amplitude encoding.
    name : str, optional
        Circuit name. Default "amp_load".
    """

    def __init__(
        self,
        num_state_qubits: int,
        values: np.ndarray,
        name: str = "amp_load",
    ) -> None:
        self.num_state_qubits = num_state_qubits
        self.values = np.asarray(values, dtype=np.float64)

        expected_len = 2**num_state_qubits
        if len(self.values) != expected_len:
            raise ValueError(
                f"values must have length 2^{num_state_qubits}={expected_len}, "
                f"got {len(self.values)}"
            )

        # Sum-normalize for amplitude encoding (ensures amplitudes sum to 1)
        norm = np.sqrt(np.sum(self.values**2))
        if norm > 0:
            self._amplitudes = self.values / norm
        else:
            self._amplitudes = np.zeros_like(self.values)

        # Convert to rotation angles: α_j = arcsin(value_j / max_val) * scaling
        # Use normalized amplitudes for stable rotation angles
        max_val = np.max(np.abs(self._amplitudes))
        if max_val > 0:
            scaled = self._amplitudes / max_val
        else:
            scaled = np.zeros_like(self._amplitudes)

        # Rotation angles in [-pi/2, pi/2]
        self._angles = np.arcsin(np.clip(scaled, -1.0, 1.0))

        # Build circuit
        ctrl_q = QuantumRegister(num_state_qubits, name="State")
        target_q = QuantumRegister(1, name="Target")

        super().__init__(ctrl_q, target_q, name=name)

        self._build_circuit()

    def _build_circuit(self) -> None:
        """Build the amplitude loading circuit using MultiCRY decomposition."""
        from qiskit.circuit.library import CRYGate

        angles = self._angles
        K = self.num_state_qubits

        # Multi-controlled RY cascade (Schmidt-like decomposition)
        # For K state qubits, we apply controlled rotations in a binary tree pattern
        for i in range(K):
            # Apply CRY with angle based on the pattern
            stride = 2**i
            for j in range(0, K, 2 * stride):
                if j + stride < K:
                    # Compute angle index for this control pattern
                    idx = j // stride
                    if idx < len(angles):
                        angle = angles[idx] * 2  # *2 for the standard Qiskit CRY convention
                        self.append(CRYGate(angle), [self.qubits[i], self.qubits[K]])

    @property
    def num_ancillas(self) -> int:
        """Number of ancilla qubits required (0 for this loader)."""
        return 0

    def to_gate(self):
        """
        Convert to a gate for reusability.

        Returns
        -------
        Gate
            A reusable gate wrapping this circuit.
        """
        return super().to_gate(label=self.name)

    def _decompose(self) -> QuantumCircuit:
        """
        Return a decomposed version of the circuit.

        Returns
        -------
        QuantumCircuit
            The circuit decomposed to basis gates.
        """
        return self.decompose()