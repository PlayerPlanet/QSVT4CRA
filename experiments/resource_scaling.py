"""
experiments/resource_scaling.py — Phase 6: Quantum Resource Scaling.

Estimates quantum resources (qubits, depth, T-count) as a function of
portfolio size K for the QSVT4CRA research run.

CLI
---
    python -m experiments.resource_scaling --n-loans 10 50 100 500 1000 --output results_scaling.png

Reference: docs/architecture.md §2 (module experiments/resource_scaling.py),
§4 (D4: pyqsp degree limits), §6 (compute: 5-10 GPU-hrs). Phase 4 modules
(PosteriorFactorCopulaLoader, QSVTRiskCircuit) are used; if not yet available,
they are stubbed per the documented interfaces in docs/architecture.md §2.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Qiskit imports — fail fast with a clear message if not installed
# ---------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import RZGate, CXGate, XGate, SXGate
except ImportError as exc:
    raise ImportError(
        "qiskit is required for resource_scaling. Install with: pip install qiskit"
    ) from exc

try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None  # type: ignore

# ---------------------------------------------------------------------------
# Local imports — use stubs where Phase 4 is incomplete
# ---------------------------------------------------------------------------
from loader.posterior_factor_copula import PosteriorFactorCopulaLoader

try:
    from qsvt.circuit import QSVTRiskCircuit
except ImportError:
    # Stub QSVTRiskCircuit if Phase 4 hasn't been implemented yet
    QSVTRiskCircuit = None  # type: ignore

try:
    from Code.multivariateGCI import MultivariateGCI_Linear
except ImportError:
    # qiskit_finance not installed — GCI unavailable
    MultivariateGCI_Linear = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper / constants
# ---------------------------------------------------------------------------
# Basis gates used for transpilation
BASIS_GATES = ["cx", "rz", "x", "sx"]

# Memory per state-vector amplitude (complex128 = 16 bytes)
BYTES_PER_COMPLEX128 = 16

# Classical simulation infeasibility threshold
INFEASIBLE_N_QUBITS = 30

# GCI max tractable K (NormalDistribution uses 2^n qubits → catastrophic for K>10)
GCI_MAX_K = 10


@dataclass
class ResourceEstimate:
    """
    Container for a single resource estimation result.

    Attributes
    ----------
    K : int
        Number of loans (portfolio size).
    n_qubits : int
        Total qubits required for the full circuit.
    n_state_qubits : int
        State qubits (K for amplitude loading).
    n_ancilla_qubits : int
        Ancilla qubits (objective + work).
    qsvt_degree : int
        QSVT polynomial degree.
    circuit_depth : int
        Total circuit depth after transpilation.
    t_count : int
        Estimated T-gate count (fault-tolerant).
    cz_count : int
        Controlled-Z gate count (proxy for two-qubit depth).
    rz_count : int
        RZ gate count.
    mcz_count : int
        Multi-controlled Z gate count.
    two_qubit_depth : int
        Depth of two-qubit (CX) gates only.
    gci_compatible : bool
        True if MultivariateGCI_Linear can handle this K.
    memory_mb_estimate : float
        State-vector memory on Aer (full state simulation).
    estimated_aer_runtime_s : float
        Estimated runtime for 1e6 shots on Aer (seconds).
    infeasible : bool
        True if classical simulation is infeasible.
    notes : str
        Additional notes or warnings.
    """

    K: int
    n_qubits: int
    n_state_qubits: int
    n_ancilla_qubits: int
    qsvt_degree: int
    circuit_depth: int
    t_count: int
    cz_count: int
    rz_count: int
    mcz_count: int
    two_qubit_depth: int
    gci_compatible: bool
    memory_mb_estimate: float
    estimated_aer_runtime_s: float
    infeasible: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        """Convert to plain dict for DataFrame / serialization."""
        d = {
            "K": self.K,
            "n_qubits": self.n_qubits,
            "n_state_qubits": self.n_state_qubits,
            "n_ancilla_qubits": self.n_ancilla_qubits,
            "qsvt_degree": self.qsvt_degree,
            "circuit_depth": self.circuit_depth,
            "t_count": self.t_count,
            "cz_count": self.cz_count,
            "rz_count": self.rz_count,
            "mcz_count": self.mcz_count,
            "two_qubit_depth": self.two_qubit_depth,
            "gci_compatible": self.gci_compatible,
            "memory_mb_estimate": self.memory_mb_estimate,
            "estimated_aer_runtime_s": self.estimated_aer_runtime_s,
            "infeasible": self.infeasible,
            "notes": self.notes,
        }
        return d


# ---------------------------------------------------------------------------
# QuantumResourceEstimator
# ---------------------------------------------------------------------------

class QuantumResourceEstimator:
    """
    Estimates quantum resources for QSVT-based credit risk circuits.

    Parameters
    ----------
    loader_factory : callable
        Factory function ``loader_factory(K: int) -> PosteriorFactorCopulaLoader``.
        Creates a ``PosteriorFactorCopulaLoader`` for the given portfolio size.
    qsvt_factory : callable, optional
        Factory function ``qsvt_factory(loader, target_loss, degree) -> QuantumCircuit``.
        Creates the full QSVT risk circuit. If None, uses the Phase 4 stub
        (``QSVTRiskCircuit``) or ``get_expected_probability_circuit`` directly.
    target_loss : float, default 0.5
        Default loss threshold for the circuit.
    default_degree : int, default 64
        Default QSVT polynomial degree.

    Example
    -------
    >>> from experiments.resource_scaling import QuantumResourceEstimator
    >>> from loader.posterior_factor_copula import PosteriorFactorCopulaLoader
    >>> def loader_factory(K):
    ...     theta = np.zeros(2*K+4, dtype=np.float32)
    ...     theta[:K] = 0.3
    ...     theta[K:2*K] = 0.02
    ...     theta[2*K+1] = 0.5
    ...     return PosteriorFactorCopulaLoader(theta, K=K)
    >>> estimator = QuantumResourceEstimator(loader_factory)
    >>> result = estimator.estimate(K=10, target_loss=0.5, degree=64)
    >>> print(result.n_qubits)
    11
    """

    def __init__(
        self,
        loader_factory: Callable[[int], PosteriorFactorCopulaLoader],
        qsvt_factory: Optional[
            Callable[[PosteriorFactorCopulaLoader, float, int], QuantumCircuit]
        ] = None,
        target_loss: float = 0.5,
        default_degree: int = 64,
    ) -> None:
        self._loader_factory = loader_factory
        self._qsvt_factory = qsvt_factory
        self._target_loss = target_loss
        self._default_degree = default_degree

    def estimate(
        self, K: int, target_loss: Optional[float] = None, degree: int = 64
    ) -> ResourceEstimate:
        """
        Estimate quantum resources for a given portfolio size K.

        Parameters
        ----------
        K : int
            Number of loans (portfolio size).
        target_loss : float, optional
            Loss threshold. Defaults to ``self._target_loss``.
        degree : int, default 64
            QSVT polynomial degree.

        Returns
        -------
        ResourceEstimate
            Dictionary-like object with all resource metrics.
        """
        target_loss = target_loss if target_loss is not None else self._target_loss

        try:
            # Build the loader
            loader = self._loader_factory(K)

            # Build the full circuit
            if self._qsvt_factory is not None:
                circuit = self._qsvt_factory(loader, target_loss, degree)
            elif QSVTRiskCircuit is not None:
                circuit = QSVTRiskCircuit(loader, target_loss, degree)
            else:
                # Fallback: use get_expected_probability_circuit directly
                from Code.circuitsCRA import get_expected_probability_circuit

                lgd = loader.lgd.tolist()
                phases = self._compute_phases_fallback(target_loss, degree, sum(lgd))
                poly = self._build_poly_fallback(degree)

                circuit, _, _ = get_expected_probability_circuit(
                    K=K,
                    uncertainity_model=loader,
                    lgd=lgd,
                    target_loss=target_loss,
                    phases=phases,
                    poly=poly,
                    threshold=0.5,
                )

            # Transpile to basis gates for realistic counts
            transpiled = transpile(
                circuit,
                basis_gates=BASIS_GATES,
                optimization_level=3,
            )

            # Extract gate counts
            gate_counts = transpiled.count_ops()

            cz_count = gate_counts.get("cx", 0)
            rz_count = gate_counts.get("rz", 0)
            mcz_count = gate_counts.get("mcz", 0)

            # Compute derived metrics
            n_qubits = transpiled.num_qubits
            n_state_qubits = K
            n_ancilla_qubits = max(n_qubits - K - 1, 0)

            # T-count estimate
            t_count = 4 * (rz_count + cz_count + mcz_count)

            # Two-qubit depth as proxy for T-depth
            two_qubit_depth = self._estimate_two_qubit_depth(transpiled)

            # Circuit depth
            circuit_depth = transpiled.depth()

            # State-vector memory
            memory_mb = self._estimate_memory_mb(n_qubits)

            # Aer runtime estimate (1e6 shots)
            runtime_s = self._estimate_aer_runtime(n_qubits, circuit_depth)

            # Check infeasibility
            infeasible = n_qubits > INFEASIBLE_N_QUBITS

            # GCI compatibility check
            gci_compatible = K <= GCI_MAX_K

            # Notes
            notes = ""
            if infeasible:
                notes += "classical-simulation-infeasible;"
            if not gci_compatible:
                notes += "GCI-NormalDistribution-would-require-2^{}qubits;".format(K)
            if degree > 128:
                notes += "pyqsp-degree-{}".format(degree)

            return ResourceEstimate(
                K=K,
                n_qubits=n_qubits,
                n_state_qubits=n_state_qubits,
                n_ancilla_qubits=n_ancilla_qubits,
                qsvt_degree=degree,
                circuit_depth=circuit_depth,
                t_count=t_count,
                cz_count=cz_count,
                rz_count=rz_count,
                mcz_count=mcz_count,
                two_qubit_depth=two_qubit_depth,
                gci_compatible=gci_compatible,
                memory_mb_estimate=memory_mb,
                estimated_aer_runtime_s=runtime_s,
                infeasible=infeasible,
                notes=notes.strip(";"),
            )

        except Exception as e:
            # Circuit construction failed (e.g., K too large for histogram approach).
            # Fall back to analytical resource estimates.
            return self._estimate_analytical(K, degree, target_loss, str(e))

    def _estimate_analytical(
        self, K: int, degree: int, target_loss: float, error_msg: str
    ) -> ResourceEstimate:
        """
        Analytical resource estimates when circuit construction fails.

        Uses known scaling laws for amplitude loading + QSVT circuits:
        - n_qubits ≈ K + 2 (K state + 1 target + ~1 ancilla)
        - circuit_depth ≈ O(K * degree)
        - t_count ≈ 4 * circuit_depth (rough estimate)

        Parameters
        ----------
        K : int
            Number of loans.
        degree : int
            QSVT degree.
        target_loss : float
            Loss threshold.
        error_msg : str
            Error message from the failed circuit construction.

        Returns
        -------
        ResourceEstimate
            Analytical resource estimate.
        """
        n_qubits = K + 2
        n_state_qubits = K
        n_ancilla_qubits = 2

        # Analytical depth estimate: O(K * degree)
        circuit_depth = K * degree * 3

        cz_count = circuit_depth // 3
        rz_count = circuit_depth
        mcz_count = degree
        t_count = 4 * (rz_count + cz_count + mcz_count)

        two_qubit_depth = circuit_depth // 4

        memory_mb = self._estimate_memory_mb(n_qubits)
        runtime_s = self._estimate_aer_runtime(n_qubits, circuit_depth)
        infeasible = n_qubits > INFEASIBLE_N_QUBITS
        gci_compatible = K <= GCI_MAX_K

        notes = "analytical-estimate;"
        if infeasible:
            notes += "classical-simulation-infeasible;"
        if not gci_compatible:
            notes += "GCI-NormalDistribution-would-require-2^{}qubits".format(K)

        return ResourceEstimate(
            K=K,
            n_qubits=n_qubits,
            n_state_qubits=n_state_qubits,
            n_ancilla_qubits=n_ancilla_qubits,
            qsvt_degree=degree,
            circuit_depth=circuit_depth,
            t_count=t_count,
            cz_count=cz_count,
            rz_count=rz_count,
            mcz_count=mcz_count,
            two_qubit_depth=two_qubit_depth,
            gci_compatible=gci_compatible,
            memory_mb_estimate=memory_mb,
            estimated_aer_runtime_s=runtime_s,
            infeasible=infeasible,
            notes=notes.strip(";"),
        )

    def compare_loader_vs_gci(self, K: int) -> dict:
        """
        Compare resource estimates for PosteriorFactorCopulaLoader and
        MultivariateGCI_Linear at the same portfolio size K.

        Parameters
        ----------
        K : int
            Number of loans.

        Returns
        -------
        dict
            Side-by-side comparison with keys:
            - ``copula``: ResourceEstimate for PosteriorFactorCopulaLoader
            - ``gci``: ResourceEstimate for MultivariateGCI_Linear
            - ``K``: the portfolio size
        """
        # PosteriorFactorCopulaLoader estimate
        copula_result = self.estimate(K, target_loss=self._target_loss, degree=self._default_degree)

        # MultivariateGCI_Linear estimate
        gci_estimate = self._estimate_gci_resources(K)

        return {
            "K": K,
            "copula": copula_result,
            "gci": gci_estimate,
        }

    def sweep(
        self, n_loans_list: list[int], degree: int = 64
    ) -> pd.DataFrame:
        """
        Run resource estimation sweep over a list of portfolio sizes.

        Parameters
        ----------
        n_loans_list : list[int]
            List of K values to sweep.
        degree : int, default 64
            QSVT polynomial degree.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per K and columns from ``ResourceEstimate.as_dict()``.
        """
        rows = []
        for K in n_loans_list:
            result = self.estimate(K, target_loss=self._target_loss, degree=degree)
            rows.append(result.as_dict())

        if rows:
            df = pd.DataFrame(rows)
        else:
            # Empty sweep — create DataFrame with correct columns
            dummy = ResourceEstimate(
                K=0, n_qubits=0, n_state_qubits=0, n_ancilla_qubits=0,
                qsvt_degree=degree, circuit_depth=0, t_count=0,
                cz_count=0, rz_count=0, mcz_count=0, two_qubit_depth=0,
                gci_compatible=True, memory_mb_estimate=0.0,
                estimated_aer_runtime_s=0.0,
            )
            df = pd.DataFrame(columns=list(dummy.as_dict().keys()))

        return df

    def plot_scaling(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Generate 4-panel scaling plot.

        Panels
        ------
        1. n_qubits vs K (log-log)
        2. circuit_depth vs K
        3. T-count vs K (log-log)
        4. estimated Aer runtime vs K (log-log)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from ``sweep()``.
        output_path : str
            Path to save the PNG figure.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_scaling. Install with: pip install matplotlib"
            )

        K_vals = df["K"].values

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Quantum Resource Scaling vs Portfolio Size K", fontsize=14)

        # Panel 1: n_qubits vs K (log-log)
        ax = axes[0, 0]
        ax.loglog(K_vals, df["n_qubits"], "o-", color="tab:blue", linewidth=2)
        ax.set_xlabel("K (loans)")
        ax.set_ylabel("n_qubits")
        ax.set_title("Qubit Count vs K")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # Add infeasible annotation
        infeasible_mask = df["infeasible"]
        if infeasible_mask.any():
            ax.axvline(x=2**INFEASIBLE_N_QUBITS, color="red", linestyle="--", alpha=0.7, label="Aer limit")
            ax.legend()

        # Panel 2: circuit_depth vs K
        ax = axes[0, 1]
        ax.plot(K_vals, df["circuit_depth"], "o-", color="tab:orange", linewidth=2)
        ax.set_xlabel("K (loans)")
        ax.set_ylabel("circuit_depth")
        ax.set_title("Circuit Depth vs K")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Panel 3: T-count vs K (log-log)
        ax = axes[1, 0]
        ax.loglog(K_vals, df["t_count"], "s-", color="tab:green", linewidth=2)
        ax.set_xlabel("K (loans)")
        ax.set_ylabel("T-count")
        ax.set_title("T-Count vs K (Fault-Tolerant Resources)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # Panel 4: estimated Aer runtime vs K (log-log)
        ax = axes[1, 1]
        ax.loglog(K_vals, df["estimated_aer_runtime_s"], "^-", color="tab:red", linewidth=2)
        ax.set_xlabel("K (loans)")
        ax.set_ylabel("estimated runtime (seconds, 1e6 shots)")
        ax.set_title("Estimated Aer Runtime vs K")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _estimate_two_qubit_depth(self, circuit: QuantumCircuit) -> int:
        """
        Estimate two-qubit gate depth by counting CX layers.

        Parameters
        ----------
        circuit : QuantumCircuit
            Transpiled circuit.

        Returns
        -------
        int
            Estimated depth of two-qubit (CX) gates.
        """
        # Build dependency graph from dag
        try:
            from qiskit.converters import circuit_to_dag

            dag = circuit_to_dag(circuit)
            # Find layers that contain two-qubit gates
            depth = 0
            qubits_in_layer = set()
            for node in dag.topological_op_nodes():
                if len(node.qargs) > 1:
                    # Multi-qubit gate — start new layer if needed
                    qids = {q.index for q in node.qargs}
                    if qubits_in_layer & qids:
                        depth += 1
                        qubits_in_layer = qids
                    else:
                        qubits_in_layer |= qids
                else:
                    pass  # Single-qubit doesn't increase depth
            return max(depth, 1)
        except Exception:
            # Fallback: estimate as total_depth * (cx_fraction)
            gate_counts = circuit.count_ops()
            total_gates = sum(gate_counts.values())
            cx_count = gate_counts.get("cx", 0)
            if total_gates == 0 or cx_count == 0:
                return 1
            # Approximate as proportion of total depth
            cx_fraction = cx_count / total_gates
            return max(int(circuit.depth() * cx_fraction), 1)

    def _estimate_memory_mb(self, n_qubits: int) -> float:
        """
        Estimate state-vector memory requirement for full Aer simulation.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.

        Returns
        -------
        float
            Memory in MB for complex128 state-vector (2^n * 16 bytes).
            Returns float('inf') if the value overflows or exceeds 1 TB.
        """
        # Guard against overflow and absurdly large values
        if n_qubits > 60:
            return float("inf")
        try:
            n_amplitudes = 2**n_qubits
            bytes_total = n_amplitudes * BYTES_PER_COMPLEX128
            mb = bytes_total / (1024**2)
            # Cap at 1 TB (approx 1e6 GB)
            if mb > 1e6:
                return float("inf")
            return mb
        except (OverflowError, MemoryError):
            return float("inf")

    def _estimate_aer_runtime(self, n_qubits: int, depth: int) -> float:
        """
        Estimate Aer simulator runtime for 1e6 shots.

        Uses empirical scaling: O(2^n_qubits * depth * n_shots) for full
        state-vector simulation. For n_qubits > 30, switches to shot-based
        sampling approximation.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        depth : int
            Circuit depth.

        Returns
        -------
        float
            Estimated runtime in seconds for 1e6 shots.
        """
        n_shots = 1_000_000

        if n_qubits <= INFEASIBLE_N_QUBITS:
            # Full state-vector simulation: O(2^n * depth * shots)
            # Rough empirical constant: ~1e-12 seconds per (amplitude * depth unit)
            amplitudes = 2**n_qubits
            # Heuristic: 2^n * depth * 1e-12 * shots
            base = amplitudes * depth * 1e-12 * n_shots
            # Add overhead for large circuits
            runtime = max(base, 1e-6)
        else:
            # Shot-based sampling: O(depth * shots) but with overhead
            # For large n, we estimate based on depth only
            runtime = depth * n_shots * 1e-9  # ~1ns per depth-unit per shot

        return float(runtime)

    def _estimate_gci_resources(self, K: int) -> ResourceEstimate:
        """
        Estimate resources for MultivariateGCI_Linear at given K.

        MultivariateGCI_Linear uses NormalDistribution which scales as 2^n
        qubits for n latent factors. For K loans, this becomes catastrophic.

        Parameters
        ----------
        K : int
            Number of loans.

        Returns
        -------
        ResourceEstimate
            Resource estimate for GCI (may be infeasible).
        """
        # GCI compatibility is determined by the theoretical scaling of
        # NormalDistribution (2^n qubits). Whether qiskit_finance is installed
        # is irrelevant for this determination.
        gci_compatible = K <= GCI_MAX_K

        # Build a GCI circuit for gate counting if qiskit_finance is available
        if gci_compatible and MultivariateGCI_Linear is not None:
            try:
                n_normal = 2  # qubits per latent factor
                sectors = 1
                p_zeros = np.full(K, 0.02, dtype=np.float32).tolist()
                rhos = np.full(K, 0.3, dtype=np.float32).tolist()
                F_list = [[1.0] * K]

                gci = MultivariateGCI_Linear(
                    n_normal=n_normal,
                    normal_max_value=2.0,
                    p_zeros=p_zeros,
                    rhos=rhos,
                    F_list=F_list,
                )
                transpiled = transpile(gci, basis_gates=BASIS_GATES, optimization_level=3)
                gate_counts = transpiled.count_ops()

                n_qubits = transpiled.num_qubits
                circuit_depth = transpiled.depth()
                cz_count = gate_counts.get("cx", 0)
                rz_count = gate_counts.get("rz", 0)
                mcz_count = gate_counts.get("mcz", 0)
                t_count = 4 * (rz_count + cz_count + mcz_count)
                two_qubit_depth = self._estimate_two_qubit_depth(transpiled)
            except Exception:
                n_qubits = 2**K
                circuit_depth = 0
                cz_count = 0
                rz_count = 0
                mcz_count = 0
                t_count = 0
                two_qubit_depth = 0
        elif gci_compatible:
            # qiskit_finance not installed but K <= GCI_MAX_K
            # Report theoretical resources based on NormalDistribution structure
            n_qubits = 2 * K  # n_normal * sectors + K (approx)
            circuit_depth = 0
            cz_count = 0
            rz_count = 0
            mcz_count = 0
            t_count = 0
            two_qubit_depth = 0
        else:
            # GCI is incompatible — report theoretical qubit count
            n_qubits = 2**K  # NormalDistribution scales as 2^n
            circuit_depth = 0
            cz_count = 0
            rz_count = 0
            mcz_count = 0
            t_count = 0
            two_qubit_depth = 0

        memory_mb = self._estimate_memory_mb(n_qubits)
        runtime_s = self._estimate_aer_runtime(n_qubits, circuit_depth) if gci_compatible else float("inf")
        infeasible = not gci_compatible or n_qubits > INFEASIBLE_N_QUBITS

        notes = ""
        if not gci_compatible:
            notes = "NormalDistribution-would-require-2^{}= qubits".format(K)
        elif K > GCI_MAX_K:
            notes = "GCI-deprecated-2^n-scaling"

        return ResourceEstimate(
            K=K,
            n_qubits=n_qubits,
            n_state_qubits=K,
            n_ancilla_qubits=0,
            qsvt_degree=self._default_degree,
            circuit_depth=circuit_depth,
            t_count=t_count,
            cz_count=cz_count,
            rz_count=rz_count,
            mcz_count=mcz_count,
            two_qubit_depth=two_qubit_depth,
            gci_compatible=gci_compatible,
            memory_mb_estimate=memory_mb,
            estimated_aer_runtime_s=runtime_s,
            infeasible=infeasible,
            notes=notes,
        )

    def _compute_phases_fallback(
        self, target_loss: float, degree: int, max_loss: float
    ):
        """Fallback phase computation if pyqsp is unavailable."""
        try:
            from pyqsp.angle_sequence import QuantumSignalProcessingPhases

            threshold = target_loss / max_loss if max_loss > 0 else 0.5
            poly = self._build_poly_fallback(degree)
            return QuantumSignalProcessingPhases(
                poly,
                signal_operator="Wx",
                method="sym_qsp",
                measurement="x",
                chebyshev_basis=True,
            )
        except ImportError:
            return [0.0] * degree

    def _build_poly_fallback(self, degree: int):
        """Fallback polynomial for phase computation."""
        poly = np.zeros(degree + 1, dtype=np.float64)
        poly[0] = -0.5
        poly[1] = 1.0
        return poly.tolist()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_default_loader_factory() -> Callable[[int], PosteriorFactorCopulaLoader]:
    """
    Build a default loader factory that creates a valid PosteriorFactorCopulaLoader
    for any K with reasonable default theta values.
    """

    def factory(K: int) -> PosteriorFactorCopulaLoader:
        theta = np.zeros(2 * K + 4, dtype=np.float32)
        theta[:K] = 0.3  # factor_loadings
        theta[K : 2 * K] = 0.02  # p_zeros
        theta[2 * K + 1] = 0.5  # rho
        return PosteriorFactorCopulaLoader(theta, K=K, max_loss=sum(np.full(K, 0.40)))

    return factory


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point for resource scaling study.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description="Quantum resource scaling study for QSVT4CRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-loans",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500, 1000],
        help="List of portfolio sizes (K values) to estimate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resource_scaling.png",
        help="Output path for the scaling plot PNG.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=64,
        help="QSVT polynomial degree.",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.5,
        help="Loss threshold for the circuit.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save results as CSV.",
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("QSVT4CRA — Phase 6: Quantum Resource Scaling")
    print("=" * 60)
    print(f"Portfolio sizes K : {args.n_loans}")
    print(f"QSVT degree       : {args.degree}")
    print(f"Target loss       : {args.target_loss}")
    print(f"Output plot       : {args.output}")
    print()

    loader_factory = _build_default_loader_factory()
    estimator = QuantumResourceEstimator(
        loader_factory=loader_factory,
        target_loss=args.target_loss,
        default_degree=args.degree,
    )

    t0 = time.time()
    df = estimator.sweep(args.n_loans, degree=args.degree)
    elapsed = time.time() - t0

    print(f"Sweep completed in {elapsed:.1f}s")
    print()

    # Print summary table
    print(df.to_string(index=False))
    print()

    # Save CSV if requested
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Results saved to {args.csv}")
        print()

    # Generate plot
    try:
        estimator.plot_scaling(df, args.output)
        print(f"Plot saved to {args.output}")
    except Exception as e:
        print(f"WARNING: Could not generate plot: {e}")

    print()
    print("Resource estimates:")
    for _, row in df.iterrows():
        infeasible_str = " [INFEASIBLE]" if row.get("infeasible", False) else ""
        gci_str = " [GCI-INCOMPATIBLE]" if not row.get("gci_compatible", True) else ""
        print(
            f"  K={row['K']:4d}: qubits={row['n_qubits']:5d}, "
            f"depth={row['circuit_depth']:8d}, T-count={row['t_count']:10d}, "
            f"memory={row['memory_mb_estimate']:12.1f} MB{gci_str}{infeasible_str}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())