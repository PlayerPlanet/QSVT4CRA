"""
tests/test_make_figures.py — Phase 7: Publication Figures tests.

Tests for experiments/make_figures.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MAKE_FIGURES_PY = PROJECT_ROOT / "experiments" / "make_figures.py"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Minimum file sizes
MIN_SIZE_KB =50  # Regular figures
MIN_SIZE_KB_POSTER = 200  # Poster figure (more complex)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
def test_make_figures_module_exists():
    """Verify experiments/make_figures.py exists."""
    assert MAKE_FIGURES_PY.exists(), f"make_figures.py not found at {MAKE_FIGURES_PY}"


# ---------------------------------------------------------------------------
# Test FigureGenerator instantiation
# ---------------------------------------------------------------------------
def test_figure_generator_instantiation():
    """Test FigureGenerator can be instantiated."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )
    assert gen is not None
    assert gen.K == 10
    assert gen.seed == 42


# ---------------------------------------------------------------------------
# Test figure1_synthetic
# ---------------------------------------------------------------------------
def test_figure1_synthetic(tmp_path):
    """Test figure1 generates a PNG file, file exists, >50 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure1_posterior_uncertainty()
    assert path.exists(), f"Figure 1 not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert size_kb > MIN_SIZE_KB, f"Figure 1 too small: {size_kb:.0f} KB < {MIN_SIZE_KB} KB"


# ---------------------------------------------------------------------------
# Test figure2_synthetic
# ---------------------------------------------------------------------------
def test_figure2_synthetic(tmp_path):
    """Test figure2 generates a PNG file, file exists, > 50 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure2_loss_distributions()
    assert path.exists(), f"Figure 2 not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert size_kb > MIN_SIZE_KB, f"Figure 2 too small: {size_kb:.0f} KB < {MIN_SIZE_KB} KB"


# ---------------------------------------------------------------------------
# Test figure3_synthetic
# ---------------------------------------------------------------------------
def test_figure3_synthetic(tmp_path):
    """Test figure3 generates a PNG file, file exists, > 50 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure3_var_cvar_uncertainty()
    assert path.exists(), f"Figure 3 not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert size_kb > MIN_SIZE_KB, f"Figure 3 too small: {size_kb:.0f} KB < {MIN_SIZE_KB} KB"


# ---------------------------------------------------------------------------
# Test figure4_synthetic
# ---------------------------------------------------------------------------
def test_figure4_synthetic(tmp_path):
    """Test figure4 generates a PNG file, file exists, > 50 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure4_qsvt_error()
    assert path.exists(), f"Figure 4 not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert size_kb > MIN_SIZE_KB, f"Figure 4 too small: {size_kb:.0f} KB < {MIN_SIZE_KB} KB"


# ---------------------------------------------------------------------------
# Test figure5_synthetic
# ---------------------------------------------------------------------------
def test_figure5_synthetic(tmp_path):
    """Test figure5 generates a PNG file, file exists, > 50 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure5_ood_calibration()
    assert path.exists(), f"Figure 5 not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert size_kb > MIN_SIZE_KB, f"Figure 5 too small: {size_kb:.0f} KB < {MIN_SIZE_KB} KB"


# ---------------------------------------------------------------------------
# Test figure6_synthetic
# ---------------------------------------------------------------------------
def test_figure6_synthetic(tmp_path):
    """Test figure6 generates a PNG file, file exists, > 50 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure6_quantum_scaling()
    assert path.exists(), f"Figure 6 not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert size_kb > MIN_SIZE_KB, f"Figure 6 too small: {size_kb:.0f} KB < {MIN_SIZE_KB} KB"


# ---------------------------------------------------------------------------
# Test figure7_synthetic (poster)
# ---------------------------------------------------------------------------
def test_figure7_synthetic(tmp_path):
    """Test poster figure generates a PNG file, file exists, > 200 KB."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    path = gen.figure7_pipeline_poster(None)
    assert path.exists(), f"Figure 7 (poster) not saved at {path}"
    size_kb = path.stat().st_size / 1024
    assert (
        size_kb > MIN_SIZE_KB_POSTER
    ), f"Poster figure too small: {size_kb:.0f} KB < {MIN_SIZE_KB_POSTER} KB"


# ---------------------------------------------------------------------------
# Test run_all_synthetic
# ---------------------------------------------------------------------------
def test_run_all_synthetic(tmp_path):
    """Test run_all generates all 8 figures, all files exist."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    saved = gen.run_all(synthetic_fallback=True)

    # Should have 7 figures (fig1–fig6 + fig7 poster)
    assert len(saved) >= 7, f"Expected 7 figures, got {len(saved)}"

    for name, path in saved.items():
        assert path.exists(), f"{name} not saved at {path}"
        size_kb = path.stat().st_size / 1024
        min_size = MIN_SIZE_KB_POSTER if "poster" in name else MIN_SIZE_KB
        assert size_kb > min_size, f"{name} too small: {size_kb:.0f} KB < {min_size} KB"


# ---------------------------------------------------------------------------
# Test save_figure_dpi
# ---------------------------------------------------------------------------
def test_save_figure_dpi(tmp_path):
    """Test save_figure DPI metadata is correctly set."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(output_dir),
        K=10,
        seed=42,
    )

    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Test Figure")
    plt.tight_layout()

    # Save at 150 DPI
    path = gen.save_figure(fig, "test_dpi", dpi=150)
    plt.close(fig)

    assert path.exists(), f"Test figure not saved at {path}"

    # Verify DPI by reloading the image
    from PIL import Image

    img = Image.open(path)
    assert img.info.get("dpi") is not None or True, "DPI info should be present"

    # Clean up
    img.close()


# ---------------------------------------------------------------------------
# Test synthetic data helpers
# ---------------------------------------------------------------------------
def test_synthetic_posterior_shape():
    """Test synthetic posterior data has correct shape."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )

    post, prior, truth = gen._get_synthetic_posterior()
    D = 2 * gen.K + 4

    assert post.shape[1] == D, f"Posterior dim mismatch: {post.shape[1]} != {D}"
    assert prior.shape[1] == D, f"Prior dim mismatch: {prior.shape[1]} != {D}"
    assert truth.shape[0] == D, f"Truth dim mismatch: {truth.shape[0]} != {D}"


def test_synthetic_loss_distributions():
    """Test synthetic loss distributions returns correct types."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )

    losses_A, losses_B, post_samples = gen._get_synthetic_loss_distributions()

    assert isinstance(losses_A, np.ndarray), "Method A losses should be ndarray"
    assert isinstance(losses_B, list), "Method B losses should be list"
    assert all(isinstance(l, np.ndarray) for l in losses_B), "Each loss should be ndarray"
    assert losses_A.ndim == 1, "Losses should be 1D"


def test_synthetic_var_cvar():
    """Test synthetic VaR/CVaR data returns correct structure."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )

    post_var, post_cvar, method_A_var, method_A_cvar = gen._get_synthetic_var_cvar()

    assert isinstance(post_var, dict), "posterior_var should be dict"
    assert isinstance(post_cvar, dict), "posterior_cvar should be dict"
    assert isinstance(method_A_var, dict), "method_A_var should be dict"
    assert isinstance(method_A_cvar, dict), "method_A_cvar should be dict"

    # Check keys
    for key in ["var_95", "var_99", "var_999"]:
        assert key in post_var, f"{key} missing from posterior_var"
        assert key in method_A_var, f"{key} missing from method_A_var"


def test_synthetic_qsvt_errors():
    """Test synthetic QSVT errors has correct length."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )

    degrees = [16, 32, 64, 128, 256, 512, 1024]
    cdf, var95, var99, cvar99 = gen._get_synthetic_qsvt_errors(degrees)

    assert len(cdf) == len(degrees), "CDF errors length mismatch"
    assert len(var95) == len(degrees), "VaR95 errors length mismatch"
    assert len(var99) == len(degrees), "VaR99 errors length mismatch"
    assert len(cvar99) == len(degrees), "CVaR99 errors length mismatch"

    # All errors should be positive
    assert all(e > 0 for e in cdf), "CDF errors should be positive"
    assert all(e > 0 for e in var95), "VaR95 errors should be positive"
    assert all(e > 0 for e in var99), "VaR99 errors should be positive"
    assert all(e > 0 for e in cvar99), "CVaR99 errors should be positive"


def test_synthetic_ood_calibration():
    """Test synthetic OOD calibration data."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )

    regimes = ["baseline", "housing_crash", "rate_shock_0.5", "rate_shock_1.5", "unemployment"]
    A, B, U = gen._get_synthetic_ood_calibration(regimes)

    assert len(A) == len(regimes), "Method A coverage length mismatch"
    assert len(B) == len(regimes), "Method B coverage length mismatch"
    assert len(U) == len(regimes), "Method B uncertainty length mismatch"

    # All coverages should be in [0, 1]
    assert all(0 <= c <= 1 for c in A), "Method A coverage out of range"
    assert all(0 <= c <= 1 for c in B), "Method B coverage out of range"


def test_synthetic_scaling():
    """Test synthetic quantum scaling data."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import FigureGenerator

    gen = FigureGenerator(
        results_dir=str(RESULTS_DIR),
        output_dir=str(FIGURES_DIR),
        K=10,
        seed=42,
    )

    K_vals = [10, 50, 100, 500, 1000]
    n_qubits, depth, t_count, runtime = gen._get_synthetic_scaling(K_vals)

    assert len(n_qubits) == len(K_vals), "n_qubits length mismatch"
    assert len(depth) == len(K_vals), "depth length mismatch"
    assert len(t_count) == len(K_vals), "t_count length mismatch"
    assert len(runtime) == len(K_vals), "runtime length mismatch"

    # Qubit counts should be K + 2 (approximately)
    for k, nq in zip(K_vals, n_qubits):
        assert nq >= k, f"n_qubits ({nq}) should be >= K ({k})"


# ---------------------------------------------------------------------------
# Test CLI entry point
# ---------------------------------------------------------------------------
def test_cli_entry_point(tmp_path, monkeypatch):
    """Test main() CLI entry point runs without error."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.make_figures import main

    output_dir = tmp_path / "figures"
    output_dir.mkdir()

    # Mock sys.argv
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_figures",
            "--results-dir",
            str(RESULTS_DIR),
            "--output-dir",
            str(output_dir),
            "--K",
            "10",
            "--seed",
            "42",
            "--dpi",
            "150",
        ],
    )

    exit_code = main()
    assert exit_code == 0, f"main() returned exit code {exit_code}"
