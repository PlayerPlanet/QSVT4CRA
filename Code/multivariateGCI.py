"""
Multivariate Gaussian Conditional Independence Models for Credit Risk.
"""
# =============================================================================
# Module attributes
# =============================================================================
# Documentation format
__docformat__ = 'NumPy'
# License type (e.g. 'GPL', 'MIT', 'Proprietary', ...)
__license__ = 'Proprietary'
# Status ('Prototype', 'Development' or 'Pre-production')
__status__ = 'Development'
# Version (0.0.x = 'Prototype', 0.x.x = 'Development', x.x.x = 'Pre-production)
__version__ = '0.1.0'
# Authors (e.g. code writers)
__author__ = ('Antonello Aita <antonello.aita@gmail.com>',
              'Emanuele Dri <emanuele.dri@polito.it>')
# Maintainer
__maintainer__ = ''
# Email of the maintainer
__email__ = ''

# =============================================================================
# Import modules
# =============================================================================
# Import general purpose module(s)

from typing import List, Union
import numpy as np
from scipy.stats.distributions import norm
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PolynomialPauliRotations, LinearPauliRotations

try:  # pragma: no cover - depends on optional qiskit-finance install
    from qiskit_finance.circuit.library import NormalDistribution  # type: ignore
except ImportError:  # pragma: no cover - exercised in lightweight CI/envs
    NormalDistribution = None  # type: ignore


def _normal_distribution_circuit(
    n_normal: int,
    mu: float,
    sigma: float,
    bounds: tuple[float, float],
) -> QuantumCircuit:
    """Return a small truncated-normal state-preparation circuit.

    The original implementation uses ``qiskit_finance.circuit.library``.
    Some modern Qiskit 2.x environments no longer ship qiskit-finance by
    default, so this local fallback keeps the K=17 Finnish-mortgage toy
    dataset buildable.  For the paper/default setting ``n_normal=2`` this is
    tiny and transpiles to ordinary single/two-qubit gates.
    """
    if NormalDistribution is not None:
        return NormalDistribution(n_normal, mu, sigma, bounds=bounds)

    grid = np.linspace(bounds[0], bounds[1], 2**n_normal, dtype=np.float64)
    probs = norm.pdf(grid, loc=mu, scale=sigma)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    amplitudes = np.sqrt(probs).astype(complex)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    qc = QuantumCircuit(n_normal, name="P(Z)")
    try:
        from qiskit.circuit.library import StatePreparation

        qc.append(StatePreparation(amplitudes), range(n_normal))
    except Exception:
        # Last-resort compatibility path.  ``initialize`` may decompose through
        # reset in some Qiskit versions, but the Heron compiler path strips to
        # backend ISA after this circuit has been converted to a gate.
        qc.initialize(amplitudes, range(n_normal))
    return qc

# =============================================================================
# =============================================================================
# MultivariateGCI_mr class
# =============================================================================
# =============================================================================
class MultivariateGCI_Poly(QuantumCircuit):
    """
    Attempt for a multivariate Gaussian Conditional 
    Independence Model for Credit Risk.
    """


    def __init__(
        self,
        n_normal: int,
        normal_max_value: float,
        p_zeros: Union[List[float], np.ndarray],
        rhos: Union[List[float], np.ndarray],
        F_list: List[float],
    ) -> None:
        """
        Args:
            n_normal: Number of qubits to represent the latent normal random variable Z
            normal_max_value: Min/max value to truncate the latent normal random variable Z
            p_zeros: Standard default probabilities for each asset
            rhos: Sensitivities of default probability of assets with respect to latent variable Z
        """
        self.n_normal = n_normal
        self.normal_max_value = normal_max_value
        self.p_zeros = p_zeros
        self.rhos = rhos
        self.sectors = len(F_list[0])
        num_qubits = n_normal*self.sectors + len(p_zeros)

        # get normal (inverse) CDF and pdf (these names are from the paper, therefore ignore
        # pylint)
        def F(x):  # pylint: disable=invalid-name
            return norm.cdf(x)

        def F_inv(x):  # pylint: disable=invalid-name
            return norm.ppf(x)

        def f(x):  # pylint: disable=invalid-name
            return norm.pdf(x)

        # create linear rotations for conditional defaults
        slopes = []
        offsets = []
        for rho, p_zero, ef in zip(rhos, p_zeros, F_list):
            psi = F_inv(p_zero) / np.sqrt(1 - rho) 
            
            # compute slope / offset
            slope_list_f_o = []
            slope_list_s_o = []
            slope_list_t_o = []
            for i in range(self.sectors):
                slope = -ef[i] / np.sqrt(1 - rho) # -np.sqrt(rho)*ef[i] / np.sqrt(1 - rho)
                slope *= f(psi) / np.sqrt(1 - F(psi)) / np.sqrt(F(psi))
                slope_list_f_o.append(slope)
                slope_list_s_o.append((slope**3)/6)
                slope_list_t_o.append((slope**5)*3/40)
            
            offset = 2 * np.arcsin(np.sqrt(F(psi)))

            # adjust for integer to normal range mapping
            # (theta(z) = slope*z + offset) and z = realization / (2^n_normal-1) * 2*normal_max_value - normal_max_value
            for i in range(self.sectors):
                offset += slope_list_f_o[i] * (-normal_max_value)
                slope_list_f_o[i] *= 2 * normal_max_value / (2 ** n_normal - 1)
                offset += slope_list_s_o[i] * ((-normal_max_value)**3)
                slope_list_s_o[i] *= 2 * normal_max_value**3 / ((2 ** n_normal - 1)**3)
                offset += slope_list_t_o[i] * ((-normal_max_value)**5)
                slope_list_t_o[i] *= 2 * normal_max_value**5 / ((2 ** n_normal - 1)**5)

            offsets += [offset]
            slopes += [(slope_list_f_o, slope_list_s_o, slope_list_t_o)]
            
        # create normal distributions        
        normal_distributions = []
        for i in range(self.sectors):
            dist = _normal_distribution_circuit(
                        n_normal,
                        0,
                        1,
                        bounds=(-normal_max_value, normal_max_value)
                    )
            normal_distributions.append(dist)

        # build circuit
        inner = QuantumCircuit(num_qubits, name="P(X)")
        for i, el in enumerate(normal_distributions):
            inner.append(el.to_gate(), list(range(i*n_normal,(i+1)*n_normal)))
        #inner.draw()

        for k, (slope, offset) in enumerate(zip(slopes, offsets)):
            for i in range(self.sectors):
                if i == 0:
                    ry = PolynomialPauliRotations(n_normal, [offset, slope[0][i], 0, slope[1][i], 0, slope[2][i]])
                else:
                    ry = PolynomialPauliRotations(n_normal, [0, slope[0][i], 0, slope[1][i], 0, slope[2][i]]) 

                qubits = list(range(i*n_normal,(i+1)*n_normal)) + [n_normal*self.sectors + k]
                inner.append(ry.to_gate(), qubits)

        super().__init__(num_qubits, name="P(X)")
        self.append(inner.to_gate(), inner.qubits)

        
# =============================================================================
# =============================================================================
# MultivariateGCI_Linear class
# =============================================================================
# =============================================================================
class MultivariateGCI_Linear(QuantumCircuit):
    """
    Attempt for a multivariate Gaussian Conditional 
    Independence Model for Credit Risk.
    With Linear Pauli Rotation
    """


    def __init__(
        self,
        n_normal: int,
        normal_max_value: float,
        p_zeros: Union[List[float], np.ndarray],
        rhos: Union[List[float], np.ndarray],
        F_list: List[float]
    ) -> None:
        """
        Args:
            n_normal: Number of qubits to represent the latent normal random variable Z
            normal_max_value: Min/max value to truncate the latent normal random variable Z
            p_zeros: Standard default probabilities for each asset
            rhos: Sensitivities of default probability of assets with respect to latent variable Z
        """
        self.n_normal = n_normal
        self.normal_max_value = normal_max_value
        self.p_zeros = p_zeros
        self.rhos = rhos
        self.sectors = len(F_list[0])
        num_qubits = n_normal*self.sectors + len(p_zeros)

        # get normal (inverse) CDF and pdf (these names are from the paper, therefore ignore
        # pylint)
        def F(x):  # pylint: disable=invalid-name
            return norm.cdf(x)

        def F_inv(x):  # pylint: disable=invalid-name
            return norm.ppf(x)

        def f(x):  # pylint: disable=invalid-name
            return norm.pdf(x)

        # create linear rotations for conditional defaults
        slopes = []
        offsets = []
        for rho, p_zero, ef in zip(rhos, p_zeros, F_list):
            psi = F_inv(p_zero) / np.sqrt(1 - rho) 

            # compute slope / offset
            slope_list_f_o = [] # first
            for i in range(self.sectors):
                slope = -ef[i] / np.sqrt(1 - rho) # -np.sqrt(rho)*ef[i] / np.sqrt(1 - rho)
                slope *= f(psi) / np.sqrt(1 - F(psi)) / np.sqrt(F(psi))
                slope_list_f_o.append(slope)

            offset = 2 * np.arcsin(np.sqrt(F(psi)))

            # adjust for integer to normal range mapping
            # (theta(z) = slope*z + offset) and z = realization / (2^n_normal-1) * 2*normal_max_value - normal_max_value
            for i in range(self.sectors):
                offset += slope_list_f_o[i] * (-normal_max_value)
                slope_list_f_o[i] *= 2 * normal_max_value / (2 ** n_normal - 1)

            offsets += [offset]
            slopes += [slope_list_f_o]

        # create normal distributions        
        normal_distributions = []
        for i in range(self.sectors):
            dist = _normal_distribution_circuit(
                        n_normal,
                        0,
                        1,
                        bounds=(-normal_max_value, normal_max_value)
                    )
            normal_distributions.append(dist)

        # build circuit
        inner = QuantumCircuit(num_qubits, name="P(X)")
        for i, el in enumerate(normal_distributions):
            inner.append(el.to_gate(), list(range(i*n_normal,(i+1)*n_normal)))

        for k, (slope, offset) in enumerate(zip(slopes, offsets)):
            for i in range(self.sectors):
                if i == 0:
                    ry = LinearPauliRotations(n_normal, slope[i], offset)
                else:
                    ry = LinearPauliRotations(n_normal, slope[i], 0) 

                qubits = list(range(i*n_normal,(i+1)*n_normal)) + [n_normal*self.sectors + k]
                inner.append(ry.to_gate(), qubits)

        super().__init__(num_qubits, name="P(X)")
        self.append(inner.to_gate(), inner.qubits)
        
        
