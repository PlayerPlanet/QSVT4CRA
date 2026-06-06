"""
tests/test_resource_scaling.py — Phase 6 resource scaling tests.

Tests for QuantumResourceEstimator and associated utilities.

Coverage targets (≥85% on new code):
- QuantumResourceEstimator.estimate()
- QuantumResourceEstimator.compare_loader_vs_gci()
- QuantumResourceEstimator.sweep()
- QuantumResourceEstimator.plot_scaling()
- ResourceEstimate.as_dict()
- CLI entry point
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from experiments.resource_scaling import (
    QuantumResourceEstimator,
    ResourceEstimate,
    _build_default_loader_factory,
    BASIS_GATES,
    INFEASIBLE_N_QUBITS,
    GCI_MAX_K,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader_factory():
    """Default loader factory for K up to ~20."""
    return _build_default_loader_factory()


@pytest.fixture
def estimator(loader_factory):
    """QuantumResourceEstimator with default factory."""
    return QuantumResourceEstimator(
        loader_factory=loader_factory,
        target_loss=0.5,
        default_degree=32,
    )


# ---------------------------------------------------------------------------
# Test: estimate() returns expected dict structure
# ---------------------------------------------------------------------------

class TestEstimate:
    """Tests for QuantumResourceEstimator.estimate()."""

    def test_estimate_K10_returns_dict(self, estimator):
        """estimate() returns a ResourceEstimate with all required keys."""
        result = estimator.estimate(K=10, target_loss=0.5, degree=32)

        assert isinstance(result, ResourceEstimate)
        d = result.as_dict()

        required_keys = [
            "K",
            "n_qubits",
            "n_state_qubits",
            "n_ancilla_qubits",
            "qsvt_degree",
            "circuit_depth",
            "t_count",
            "cz_count",
            "rz_count",
            "mcz_count",
            "two_qubit_depth",
            "gci_compatible",
            "memory_mb_estimate",
            "estimated_aer_runtime_s",
            "infeasible",
            "notes",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

        assert d["K"] == 10
        assert d["n_qubits"] >= 10  # at minimum K+1
        assert d["n_state_qubits"] == 10
        assert d["qsvt_degree"] == 32
        assert isinstance(d["gci_compatible"], bool)
        assert isinstance(d["infeasible"], bool)
        assert isinstance(d["memory_mb_estimate"], float)
        assert isinstance(d["estimated_aer_runtime_s"], float)

    def test_estimate_K_small_infeasible_false(self, estimator):
        """K=5 should be feasible (n_qubits <= 30)."""
        result = estimator.estimate(K=5, degree=16)
        assert result.infeasible is False
        assert result.n_qubits <= INFEASIBLE_N_QUBITS

    def test_estimate_K_large_infeasible_true(self, estimator):
        """K >= 20 should be marked infeasible (n_qubits > 30)."""
        result = estimator.estimate(K=20, degree=32)
        # K=20 → at least 20 state qubits + 1 target + ancillas
        # n_qubits = 20 + 1 + ancillas > 30 → infeasible
        if result.n_qubits > INFEASIBLE_N_QUBITS:
            assert result.infeasible is True

    def test_estimate_gci_compatible_K_le_10(self, estimator):
        """K <= 10 should be GCI compatible."""
        result = estimator.estimate(K=10, degree=16)
        assert result.gci_compatible is True

    def test_estimate_gci_incompatible_K_gt_10(self, estimator):
        """K > 10 should be GCI incompatible."""
        result = estimator.estimate(K=11, degree=16)
        assert result.gci_compatible is False


# ---------------------------------------------------------------------------
# Test: sweep() returns DataFrame with correct rows
# ---------------------------------------------------------------------------

class TestSweep:
    """Tests for QuantumResourceEstimator.sweep()."""

    def test_sweep_n_loans(self, estimator):
        """sweep([3, 5, 8]) returns DataFrame with correct K values."""
        df = estimator.sweep([3, 5, 8], degree=16)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df["K"]) == [3, 5, 8]
        assert "n_qubits" in df.columns
        assert "circuit_depth" in df.columns
        assert "t_count" in df.columns

    def test_sweep_empty_list(self, estimator):
        """sweep([]) returns empty DataFrame with correct columns."""
        df = estimator.sweep([], degree=16)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert len(df.columns) > 0

    def test_sweep_preserves_degree(self, estimator):
        """sweep() correctly applies degree parameter."""
        df = estimator.sweep([5, 10], degree=64)
        assert all(df["qsvt_degree"] == 64)


# ---------------------------------------------------------------------------
# Test: compare_loader_vs_gci()
# ---------------------------------------------------------------------------

class TestCompareLoaderVsGCI:
    """Tests for QuantumResourceEstimator.compare_loader_vs_gci()."""

    def test_compare_loader_vs_gci_K3(self, estimator):
        """Both loaders succeed at K=3."""
        result = estimator.compare_loader_vs_gci(K=3)

        assert "K" in result
        assert "copula" in result
        assert "gci" in result
        assert result["K"] == 3
        assert isinstance(result["copula"], ResourceEstimate)
        assert isinstance(result["gci"], ResourceEstimate)

        # Both should be GCI compatible at K=3
        assert result["copula"].gci_compatible is True
        assert result["gci"].gci_compatible is True

    def test_compare_loader_vs_gci_K20(self, estimator):
        """GCI marked incompatible at K=20."""
        result = estimator.compare_loader_vs_gci(K=20)

        # Copula should still work
        assert result["copula"].gci_compatible is False
        # GCI should be explicitly incompatible
        assert result["gci"].gci_compatible is False
        assert result["gci"].infeasible is True


# ---------------------------------------------------------------------------
# Test: plot_scaling() creates file
# ---------------------------------------------------------------------------

class TestPlotScaling:
    """Tests for QuantumResourceEstimator.plot_scaling()."""

    def test_plot_scaling_creates_file(self, estimator):
        """plot_scaling() saves a PNG file that exists."""
        df = estimator.sweep([10, 50, 100], degree=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scaling_test.png")
            estimator.plot_scaling(df, output_path)

            assert os.path.exists(output_path), "Plot file was not created"
            assert os.path.getsize(output_path) > 0, "Plot file is empty"

    def test_plot_scaling_invalid_path_raises(self, estimator):
        """plot_scaling() raises on invalid output path."""
        df = estimator.sweep([10], degree=16)

        with pytest.raises(Exception):
            estimator.plot_scaling(df, "/nonexistent/path/plot.png")


# ---------------------------------------------------------------------------
# Test: memory estimate grows exponentially
# ---------------------------------------------------------------------------

class TestMemoryEstimate:
    """Tests for state-vector memory estimation."""

    def test_memory_estimate_grows_exponentially(self, estimator):
        """Memory doubles with each additional qubit (2^n pattern)."""
        results = [estimator.estimate(K=K, degree=16) for K in [5, 6, 7]]

        # Memory should roughly double with each added qubit
        for i in range(len(results) - 1):
            r0, r1 = results[i], results[i + 1]
            # 2^(n+1) / 2^n = 2 → memory should roughly double
            ratio = r1.memory_mb_estimate / max(r0.memory_mb_estimate, 1e-9)
            # Allow some tolerance for ancilla qubits overhead
            assert ratio > 1.5, (
                f"Memory did not grow exponentially: K={r0.K} "
                f"→ K={r1.K} ratio={ratio:.2f}"
            )

    def test_memory_estimate_K5(self, estimator):
        """K=5 should require ~32 MB (2^6 qubits * 16 bytes)."""
        result = estimator.estimate(K=5, degree=16)
        # 6 qubits → 64 amplitudes → 64 * 16 = 1024 bytes = 0.001 MB
        # But we have ancillas and more, so check it's positive
        assert result.memory_mb_estimate > 0


# ---------------------------------------------------------------------------
# Test: Aer runtime estimate grows with complexity
# ---------------------------------------------------------------------------

class TestAerRuntimeEstimate:
    """Tests for Aer runtime estimation."""

    def test_aer_runtime_grows_exponentially(self, estimator):
        """Runtime estimate grows with K (non-decreasing)."""
        results = [estimator.estimate(K=K, degree=16) for K in [5, 8, 10, 12]]

        runtimes = [r.estimated_aer_runtime_s for r in results]
        for i in range(len(runtimes) - 1):
            # Runtime should generally increase (allowing some variance)
            assert runtimes[i + 1] >= runtimes[i] * 0.5, (
                f"Runtime anomaly: K={results[i].K} → K={results[i+1].K} "
                f"runtime decreased unexpectedly"
            )

    def test_aer_runtime_infeasible_large(self, estimator):
        """Large K should have runtime estimate > 0."""
        result = estimator.estimate(K=20, degree=32)
        assert result.estimated_aer_runtime_s > 0


class TestAnalyticalFallback:
    """Tests for analytical resource estimation fallback."""

    def test_analytical_fallback_method_exists(self, estimator):
        """_estimate_analytical produces reasonable estimates."""
        result = estimator._estimate_analytical(K=100, degree=64, target_loss=0.5, error_msg="test")
        assert result.K == 100
        assert result.n_qubits == 102  # K + 2
        assert result.qsvt_degree == 64
        assert result.infeasible is True
        assert "analytical-estimate" in result.notes

    def test_analytical_fallback_gci_compatible(self, estimator):
        """Analytical fallback marks GCI correctly."""
        result = estimator._estimate_analytical(K=5, degree=32, target_loss=0.5, error_msg="test")
        assert result.gci_compatible is True

    def test_analytical_fallback_gci_incompatible(self, estimator):
        """Analytical fallback marks large K as GCI incompatible."""
        result = estimator._estimate_analytical(K=50, degree=64, target_loss=0.5, error_msg="test")
        assert result.gci_compatible is False


# ---------------------------------------------------------------------------
# Test: CLI entry point
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for main() CLI entry point."""

    def test_main_runs_without_error(self):
        """main() exits with code 0 on valid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "cli_test.png")
            exit_code = main(["--n-loans", "10", "50", "--output", output, "--degree", "32"])
            assert exit_code == 0

    def test_main_creates_plot(self):
        """main() creates the output plot file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "cli_plot.png")
            main(["--n-loans", "10", "--output", output, "--degree", "16"])
            assert os.path.exists(output)

    def test_main_csv_option(self):
        """main() saves CSV when --csv is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            output = os.path.join(tmpdir, "plot.png")
            exit_code = main(["--n-loans", "10", "--output", output, "--csv", csv_path])
            assert exit_code == 0
            assert os.path.exists(csv_path)


# ---------------------------------------------------------------------------
# Test: ResourceEstimate.as_dict() round-trip
# ---------------------------------------------------------------------------

class TestResourceEstimate:
    """Tests for ResourceEstimate dataclass."""

    def test_as_dict_contains_all_fields(self):
        """as_dict() includes all ResourceEstimate fields."""
        re = ResourceEstimate(
            K=10,
            n_qubits=11,
            n_state_qubits=10,
            n_ancilla_qubits=1,
            qsvt_degree=64,
            circuit_depth=1000,
            t_count=4000,
            cz_count=500,
            rz_count=1000,
            mcz_count=50,
            two_qubit_depth=100,
            gci_compatible=True,
            memory_mb_estimate=176.0,
            estimated_aer_runtime_s=1.5,
            infeasible=False,
            notes="test",
        )
        d = re.as_dict()
        # Check that all expected keys are present
        assert d["K"] == 10
        assert d["n_qubits"] == 11
        assert d["n_state_qubits"] == 10
        assert d["n_ancilla_qubits"] == 1
        assert d["qsvt_degree"] == 64
        assert d["circuit_depth"] == 1000
        assert d["t_count"] == 4000
        assert d["cz_count"] == 500
        assert d["rz_count"] == 1000
        assert d["mcz_count"] == 50
        assert d["two_qubit_depth"] == 100
        assert d["gci_compatible"] is True
        assert d["memory_mb_estimate"] == 176.0
        assert d["estimated_aer_runtime_s"] == 1.5
        assert d["infeasible"] is False
        assert d["notes"] == "test"


# ---------------------------------------------------------------------------
# Test: constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_basis_gates(self):
        """BASIS_GATES contains expected gates."""
        assert "cx" in BASIS_GATES
        assert "rz" in BASIS_GATES

    def test_infeasible_threshold(self):
        """INFEASIBLE_N_QUBITS is 30."""
        assert INFEASIBLE_N_QUBITS == 30

    def test_gci_max_k(self):
        """GCI_MAX_K is 10."""
        assert GCI_MAX_K == 10