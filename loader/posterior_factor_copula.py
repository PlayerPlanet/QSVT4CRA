"""
PosteriorFactorCopulaLoader — replaces MultivariateGCI_* loaders.

Uses amplitude loading (NOT NormalDistribution) to encode the loss
distribution for a posterior sample θ. Compatible with
get_expected_probability_circuit(uncertainity_model=loader, ...).

Architecture: D3 (docs/architecture.md §3)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit import QuantumRegister

from loader.amplitude_loader import AmplitudeLoader


def _mapping(decimal_number: int, lgd: np.ndarray) -> float:
    """
    Compute total portfolio loss for a binary default pattern.

    Parameters
    ----------
    decimal_number : int
        Binary pattern as integer (0 to 2^K-1).
    lgd : np.ndarray, shape (K,)
        Loss-given-default for each loan.

    Returns
    -------
    float
        Total loss = sum(lgd[i] for bits[i]=1).
    """
    K = len(lgd)
    binary = format(decimal_number, f"0{K}b")
    losses = [lgd[i] for i, bit in enumerate(binary) if bit == "1"]
    return float(sum(losses))


class PosteriorFactorCopulaLoader(QuantumCircuit):
    """
    Quantum circuit loader that encodes a factor-copula loss distribution.

    Replaces MultivariateGCI_Linear and MultivariateGCI_Poly using
    amplitude loading instead of NormalDistribution.

    For a posterior sample θ, this circuit encodes the probability
    distribution P(Loss = l_j) over all2^K scenarios j as amplitudes.

    Parameters
    ----------
    theta : np.ndarray, shape (D,)
        Posterior sample parameter vector.
        Layout: [factor_loadings:K, p_zeros:K, tail_dep, rho, nu, spare]
    K : int, default 10
        Number of loans (state qubits = K).
    max_loss : float, default 1.0
        Maximum portfolio loss for normalization.
    name : str, optional
        Circuit name. Default "PFC".

    Attributes
    ----------
    num_qubits : int
        K state qubits + 1 objective qubit.

    Notes
    -----
    - Uses GaussianFactorCopula for scenario probability computation
    - Amplitude loading avoids the O(2^n) qubit scaling of NormalDistribution
    - Compatible with get_expected_probability_circuit(uncertainity_model=...)
    """

    def __init__(
        self,
        theta: np.ndarray,
        K: int = 10,
        max_loss: float = 1.0,
        name: str = "PFC",
    ) -> None:
        self._theta = np.asarray(theta, dtype=np.float32)
        self._K = K
        self._max_loss = max_loss

        # LGD array (fixed at 0.40 for apartment loans)
        self._lgd = np.full(K, 0.40, dtype=np.float32)

        # Number of scenarios = 2^K
        self._n_scenarios = 2**K

        # Build quantum registers
        self._state_q = QuantumRegister(self._K, name="state")
        self._target_q = QuantumRegister(1, name="target")
        super().__init__(self._state_q, self._target_q, name=name)
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Build the amplitude loading circuit for loss distribution."""
        K = self._K
        lgd = self._lgd
        max_loss = self._max_loss
        theta = self._theta

        # Generate scenario probabilities using Gaussian factor copula
        from copula.gaussian import GaussianFactorCopula

        copula = GaussianFactorCopula(K=K, seed=42)
        U, _ = copula.sample(theta, n_samples=10000)

        # Compute scenario probabilities from uniform marginals
        # For each scenario j, P(default pattern j) = product over loans of P(default_i | U_i)
        # We approximate by histogram over2^K bins
        scenario_probs = self._compute_scenario_probs(U, K)

        # Compute loss for each scenario
        losses = np.array(
            [_mapping(j, lgd) for j in range(self._n_scenarios)],
            dtype=np.float32
        )

        # Normalize losses to [0, 1] range for amplitude encoding
        if max_loss > 0:
            normalized_losses = losses / max_loss
        else:
            normalized_losses = np.zeros_like(losses)

        # Create amplitude loader for the loss distribution
        # Use the scenario probabilities as weights for amplitude encoding
        amplitudes = np.sqrt(np.clip(scenario_probs, 1e-10, 1.0))
        amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))

        # Build amplitude loading circuit
        amp_loader = AmplitudeLoader(K, amplitudes, name="amp_load")
        self.append(amp_loader.to_gate(), self.qubits)

    def _compute_scenario_probs(self, U: np.ndarray, K: int) -> np.ndarray:
        """
        Compute scenario probabilities from uniform marginals.

        Parameters
        ----------
        U : np.ndarray, shape (n_samples, K)
            Uniform marginals from copula.
        K : int
            Number of loans.

        Returns
        -------
        probs : np.ndarray, shape (2^K,)
            Probability for each scenario.
        """
        n_samples = U.shape[0]
        n_scenarios = 2**K

        # Convert marginals to binary default patterns
        # Default threshold: p_zero from theta
        p_zeros = self._theta[K : 2 * K]

        # Binary default patterns
        defaults = (U < p_zeros[None, :]).astype(np.int8)

        # Convert each sample's default pattern to decimal
        weights = 2 ** np.arange(K, dtype=np.int32)
        scenario_indices = np.dot(defaults, weights).astype(np.int32)

        # Histogram to get scenario counts
        probs = np.bincount(scenario_indices, minlength=n_scenarios)
        probs = probs.astype(np.float64)
        probs = probs / n_samples

        return probs.astype(np.float32)

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (K state + 1 target)."""
        return self._K + 1

    @property
    def K(self) -> int:
        """Number of state qubits (loans)."""
        return self._K

    @property
    def lgd(self) -> np.ndarray:
        """LGD array."""
        return self._lgd.copy()

    def _decompose(self) -> QuantumCircuit:
        """
        Return decomposed circuit.

        Returns
        -------
        QuantumCircuit
            Circuit decomposed to basis gates.
        """
        return self.decompose()
