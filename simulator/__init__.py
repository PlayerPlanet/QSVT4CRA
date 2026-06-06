"""
Simulator package for QSVT4CRA research run.
Provides forward models (θ → observations) in JAX and NumPy backends.
"""
from .forward import ForwardSimulator, JAXForwardSimulator, NumPyForwardSimulator

__all__ = [
    "ForwardSimulator",
    "JAXForwardSimulator",
    "NumPyForwardSimulator",
]
