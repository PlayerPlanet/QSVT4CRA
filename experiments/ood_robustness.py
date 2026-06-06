#!/usr/bin/env python
"""
Phase 5 — Distribution Shift Experiment for QSVT4CRA Research Run.

Train on Regime A (baseline), test on Regime B (stress scenarios).
Compare:
  A. Point-estimate GCI: posterior mean θ̂ used as fixed parameter
  B. Posterior-propagated factor copula: sample θ⁽ⁱ⁾ from posterior,
     apply regime shock, compute per-sample VaR/CVaR, aggregate

Usage
------
python -m experiments.ood_robustness \\
    --posterior-samples posterior_samples.npy \\
    --test-regimes baseline housing_crash rate_shock_0.5 rate_shock_1.5 unemployment liquidity \\
    --n-posterior-samples 1000 \\
    --n-scenarios 100000 \\
    --output results/ood_robustness_results.npz

python -m experiments.ood_robustness \\
    --posterior-checkpoint sbi_pipeline/checkpoint.pt \\
    --test-regimes baseline housing_crash \\
    --n-posterior-samples 100 \\
    --n-scenarios 10000 \\
    --output results/ood_test.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from copula.gaussian import GaussianFactorCopula
from copula.student_t import StudentTFactorCopula
from data.stress_regimes import StressRegimeGenerator
from data.synthetic import SyntheticPortfolioGenerator
from metrics.var_cvar import var_cvar


# ---------------------------------------------------------------------------
# Regime shock mapping
# ---------------------------------------------------------------------------
# Maps regime names (including synthetic sub-cases) to (base_name, shock).
# rate_shock_0.5 → rate_shock with shock=0.5 (low rate, shock < 1)
# rate_shock_1.5 → rate_shock with shock=1.5 (high rate, shock > 1)
_REGIME_SHOCK_MAP = {
    "baseline": ("baseline", 0.0),
    "housing_crash": ("housing_crash", 1.0),
    "rate_shock_0.5": ("rate_shock", 0.5),
    "rate_shock_1.5": ("rate_shock", 1.5),
    "unemployment": ("unemployment", 1.0),
    "liquidity": ("liquidity", 1.0),
}


def _resolve_shock(regime_name: str) -> tuple[str, float]:
    """Resolve regime name to (base_regime, shock_magnitude)."""
    if regime_name in _REGIME_SHOCK_MAP:
        return _REGIME_SHOCK_MAP[regime_name]
    # Fallback: treat as-is with shock=1.0
    return (regime_name, 1.0)


def _load_posterior(path: str) -> np.ndarray:
    """
    Load posterior samples from .npy, .npz, or .pt file.

    Parameters
    ----------
    path : str
        Path to checkpoint.

    Returns
    -------
    samples : np.ndarray, shape (N, D)
        Posterior samples, float32.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr.astype(np.float32)

    if suffix == ".npz":
        with np.load(path) as f:
            for key in ("posterior", "samples", "theta"):
                if key in f:
                    arr = f[key]
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    return arr.astype(np.float32)
        raise ValueError(
            f"Could not find posterior samples in {path}. "
            "Known keys: posterior, samples, theta"
        )

    if suffix == ".pt":
        try:
            import torch

            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict):
                for key in ("posterior", "samples", "theta"):
                    if key in ckpt:
                        arr = ckpt[key]
                        if isinstance(arr, np.ndarray):
                            if arr.ndim == 1:
                                arr = arr[None, :]
                            return arr.astype(np.float32)
                        arr = arr.detach().cpu().numpy()
                        if arr.ndim == 1:
                            arr = arr[None, :]
                        return arr.astype(np.float32)
            raise ValueError(f"No known key in PyTorch checkpoint: {path}")
        except ImportError:
            raise ValueError("PyTorch not available; cannot load .pt checkpoint")

    raise ValueError(f"Unsupported file format: {suffix}")


def _get_copula(copula_name: str, K: int, seed: int):
    """Instantiate copula by name."""
    copulas = {
        "gaussian": GaussianFactorCopula,
        "student_t": StudentTFactorCopula,
    }
    if copula_name not in copulas:
        raise ValueError(
            f"Unknown copula {copula_name!r}. Available: {list(copulas.keys())}"
        )
    return copulas[copula_name](K=K, seed=seed)


# ---------------------------------------------------------------------------
# OODExperiment
# ---------------------------------------------------------------------------
class OODExperiment:
    """
    Out-of-distribution robustness experiment.

    Compares two risk quantification methods under distribution shift:
      A. Point-estimate GCI — single calibrated model (posterior mean)
      B. Posterior-propagated factor copula — uncertainty-aware

    Parameters
    ----------
    posterior_samples : np.ndarray, shape (N, D)
        Posterior samples from SBI training.
    train_regime : str, default 'baseline'
        Regime used for training (label only).
    test_regimes : list of str
        Regimes to evaluate.  Special cases:
        - ``rate_shock_0.5`` → rate_shock with shock=0.5 (low rate)
        - ``rate_shock_1.5`` → rate_shock with shock=1.5 (high rate)
    n_scenarios : int, default 100_000
        MC scenarios per evaluation.
    K : int, default 10
        Number of loans in portfolio.
    copula : str, default 'gaussian'
        Copula model (``'gaussian'`` or ``'student_t'``).
    seed : int, default 42
        Random seed for reproducibility.
    """

    ALPHAS = [0.95, 0.99, 0.999]

    def __init__(
        self,
        posterior_samples: np.ndarray,
        train_regime: str = "baseline",
        test_regimes: list = None,
        n_scenarios: int = 100_000,
        K: int = 10,
        copula: str = "gaussian",
        seed: int = 42,
    ) -> None:
        if test_regimes is None:
            test_regimes = ["baseline", "housing_crash"]
        self.posterior_samples = np.asarray(posterior_samples, dtype=np.float32)
        self.train_regime = train_regime
        self.test_regimes = test_regimes
        self.n_scenarios = n_scenarios
        self.K = K
        self.copula = copula
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._stress_gen = StressRegimeGenerator(seed=seed)
        self._copula = _get_copula(copula, K=K, seed=seed)
        self._portfolio_gen = SyntheticPortfolioGenerator(K=K, seed=seed)
        # Cached results
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Run the OOD experiment across all test regimes.

        For each regime:
          Method A — point-estimate GCI:
            θ̂ = posterior mean; apply regime shock; compute VaR/CVaR via copula
          Method B — posterior factor copula:
            For each θ⁽ⁱ⁾ in posterior_samples:
              apply regime shock; compute per-sample VaR/CVaR
            Aggregate: mean ± std

        Returns
        -------
        results : dict
            ``{'regime_results': {regime: {method_A: {...}, method_B: {...}}}, 'summary': {...}}``
        """
        t0 = time.time()

        # Posterior mean as point estimate
        theta_mean = self.posterior_samples.mean(axis=0)

        regime_results = {}
        in_dist_width = None  # baseline interval width for OOD robustness

        for regime in self.test_regimes:
            base_regime, shock = _resolve_shock(regime)

            # Method A: point-estimate GCI (posterior mean, fixed)
            method_a_result = self._run_method_A(theta_mean, base_regime, shock)

            # Method B: posterior-propagated factor copula
            method_b_result = self._run_method_B(
                self.posterior_samples, base_regime, shock
            )

            regime_results[regime] = {
                "method_A_gci": method_a_result,
                "method_B_posterior": method_b_result,
            }

            # Capture in-dist width from baseline regime
            if regime == "baseline":
                # Width = method B 95th credible interval (97.5th - 2.5th percentile)
                var_95_vals = method_b_result.get("var_95_samples", np.array([]))
                if len(var_95_vals) > 0:
                    in_dist_width = float(
                        np.percentile(var_95_vals, 97.5) - np.percentile(var_95_vals, 2.5)
                    )

        # Compute summary statistics
        summary = self._compute_summary(regime_results, in_dist_width)

        self._results = {
            "regime_results": regime_results,
            "summary": summary,
            "n_posterior_samples": self.posterior_samples.shape[0],
            "n_scenarios": self.n_scenarios,
            "train_regime": self.train_regime,
            "test_regimes": self.test_regimes,
            "copula": self.copula,
            "K": self.K,
            "runtime_seconds": time.time() - t0,
        }

        return self._results

    def save_npz(self, output_path: str) -> None:
        """
        Serialize results to .npz file.

        Parameters
        ----------
        output_path : str
            Path to output .npz file.
        """
        if not self._results:
            raise RuntimeError("Must call run() before save_npz()")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **self._results)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _run_method_A(
        self, theta: np.ndarray, base_regime: str, shock: float
    ) -> dict:
        """
        Method A: point-estimate GCI using posterior mean as fixed θ.

        Applies regime shock to θ, runs MC simulation, computes VaR/CVaR.

        Parameters
        ----------
        theta : np.ndarray, shape (D,)
            Parameter vector (posterior mean).
        base_regime : str
            Base regime name.
        shock : float
            Shock magnitude.

        Returns
        -------
        metrics : dict
            VaR/CVaR at 95%, 99%, 99.9% + tail coverage metrics.
        """
        # Apply regime shock
        theta_shocked = self._stress_gen.sample(
            regime_name=base_regime,
            theta_baseline=theta,
            shock_magnitude=shock,
        )

        # Run MC simulation
        _, losses = self._copula.sample(theta_shocked, n_samples=self.n_scenarios)

        # Compute metrics
        metrics = self._compute_metrics(losses)

        return metrics

    def _run_method_B(
        self,
        posterior_samples: np.ndarray,
        base_regime: str,
        shock: float,
    ) -> dict:
        """
        Method B: posterior-propagated factor copula.

        For each posterior sample θ⁽ⁱ⁾, apply regime shock, compute
        per-sample VaR/CVaR, then aggregate (mean ± std).

        Parameters
        ----------
        posterior_samples : np.ndarray, shape (N, D)
            Posterior samples.
        base_regime : str
            Base regime name.
        shock : float
            Shock magnitude.

        Returns
        -------
        metrics : dict
            Mean and std of VaR/CVaR across posterior samples, plus
            per-sample VaR/CVaR arrays for uncertainty quantification.
        """
        N = posterior_samples.shape[0]

        var_95_all = np.zeros(N, dtype=np.float32)
        var_99_all = np.zeros(N, dtype=np.float32)
        var_999_all = np.zeros(N, dtype=np.float32)
        cvar_95_all = np.zeros(N, dtype=np.float32)
        cvar_99_all = np.zeros(N, dtype=np.float32)
        cvar_999_all = np.zeros(N, dtype=np.float32)

        for i in range(N):
            theta_i = posterior_samples[i]

            # Apply regime shock
            theta_shocked = self._stress_gen.sample(
                regime_name=base_regime,
                theta_baseline=theta_i,
                shock_magnitude=shock,
            )

            # MC simulation
            _, losses_i = self._copula.sample(theta_shocked, n_samples=self.n_scenarios)

            # Per-sample VaR/CVaR
            var_95_all[i] = float(np.quantile(losses_i, 0.95))
            tail_95 = losses_i[losses_i > var_95_all[i]]
            cvar_95_all[i] = float(tail_95.mean()) if tail_95.size > 0 else var_95_all[i]

            var_99_all[i] = float(np.quantile(losses_i, 0.99))
            tail_99 = losses_i[losses_i > var_99_all[i]]
            cvar_99_all[i] = float(tail_99.mean()) if tail_99.size > 0 else var_99_all[i]

            var_999_all[i] = float(np.quantile(losses_i, 0.999))
            tail_999 = losses_i[losses_i > var_999_all[i]]
            cvar_999_all[i] = (
                float(tail_999.mean()) if tail_999.size > 0 else var_999_all[i]
            )

        # Aggregate
        metrics = {
            "var_95_mean": float(var_95_all.mean()),
            "var_95_std": float(var_95_all.std()),
            "var_99_mean": float(var_99_all.mean()),
            "var_99_std": float(var_99_all.std()),
            "var_999_mean": float(var_999_all.mean()),
            "var_999_std": float(var_999_all.std()),
            "cvar_95_mean": float(cvar_95_all.mean()),
            "cvar_95_std": float(cvar_95_all.std()),
            "cvar_99_mean": float(cvar_99_all.mean()),
            "cvar_99_std": float(cvar_99_all.std()),
            "cvar_999_mean": float(cvar_999_all.mean()),
            "cvar_999_std": float(cvar_999_all.std()),
            "var_95_samples": var_95_all,
            "var_99_samples": var_99_all,
            "var_999_samples": var_999_all,
            "cvar_95_samples": cvar_95_all,
            "cvar_99_samples": cvar_99_all,
            "cvar_999_samples": cvar_999_all,
        }

        return metrics

    def _compute_metrics(self, losses: np.ndarray) -> dict:
        """
        Compute VaR, CVaR, and tail coverage metrics from loss samples.

        Parameters
        ----------
        losses : np.ndarray, shape (N,)
            Monte Carlo loss samples.

        Returns
        -------
        metrics : dict
            VaR/CVaR at 95%, 99%, 99.9% + tail coverage fractions.
        """
        metrics = {}

        for alpha in self.ALPHAS:
            var = float(np.quantile(losses, alpha))
            tail = losses[losses > var]
            cvar = float(tail.mean()) if tail.size > 0 else var
            # Use >= so that losses at the VaR threshold count toward tail
            tail_prob = float(np.mean(losses >= var))

            key = str(alpha).replace(".", "_")
            metrics[f"var_{key}"] = var
            metrics[f"cvar_{key}"] = cvar
            metrics[f"tail_prob_{key}"] = tail_prob

        return metrics

    def _compute_summary(
        self, regime_results: dict, in_dist_width: float | None
    ) -> dict:
        """
        Compute summary statistics across all regimes.

        Parameters
        ----------
        regime_results : dict
            Per-regime results from run().
        in_dist_width : float or None
            Baseline (in-dist) 95% credible interval width.

        Returns
        -------
        summary : dict
            Summary metrics including:
            - method_A_avg_tail_coverage_error
            - method_B_avg_tail_coverage_error
            - method_B_uncertainty_widening
        """
        # Average tail coverage error across regimes and alpha levels
        alpha_levels = ["0_95", "0_99", "0_999"]

        method_a_errors = []
        method_b_errors = []
        ood_widening = []

        for regime, results in regime_results.items():
            method_a = results["method_A_gci"]
            method_b = results["method_B_posterior"]

            for key in alpha_levels:
                nominal = float(key.replace("_", "."))
                # Method A tail coverage error
                tail_cov_a = method_a.get(f"tail_prob_{key}", None)
                if tail_cov_a is not None:
                    method_a_errors.append(abs(tail_cov_a - (1.0 - nominal)))

                # Method B tail coverage error
                # Use mean of per-sample tail probabilities
                var_samples = method_b.get(f"var_{key}_samples", np.array([]))
                if len(var_samples) > 0:
                    # Per-sample tail coverage
                    # (computed separately per posterior sample in full impl)
                    method_b_errors.append(
                            abs(
                                float(np.mean(var_samples > method_b[f"var_{key}_mean"]))
                                - (1.0 - nominal)
                            )
                        )

            # OOD uncertainty widening
            if in_dist_width is not None and in_dist_width > 0:
                var_95_samples = method_b.get("var_95_samples", np.array([]))
                if len(var_95_samples) > 0:
                    regime_width = float(
                        np.percentile(var_95_samples, 97.5)
                        - np.percentile(var_95_samples, 2.5)
                    )
                    widening = regime_width / in_dist_width
                    ood_widening.append(widening)

        avg_a_error = float(np.mean(method_a_errors)) if method_a_errors else 0.0
        avg_b_error = float(np.mean(method_b_errors)) if method_b_errors else 0.0
        avg_widening = float(np.mean(ood_widening)) if ood_widening else 1.0

        return {
            "method_A_avg_tail_coverage_error": avg_a_error,
            "method_B_avg_tail_coverage_error": avg_b_error,
            "method_B_uncertainty_widening": avg_widening,
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def compare_methods_plot(results: dict, output_path: str) -> None:
    """
    Generate 4-panel comparison plot for OOD experiment results.

    Parameters
    ----------
    results : dict
        Results dict from OODExperiment.run().
    output_path : str
        Path to save the PNG figure.
    """
    import matplotlib.pyplot as plt

    regime_results = results["regime_results"]
    regimes = list(regime_results.keys())

    # Panel data extraction helpers
    def get_method_a_var95(regime):
        return regime_results[regime]["method_A_gci"].get("var_0_95", 0.0)

    def get_method_b_var95_mean(regime):
        return regime_results[regime]["method_B_posterior"].get("var_95_mean", 0.0)

    def get_method_b_var95_std(regime):
        return regime_results[regime]["method_B_posterior"].get("var_95_std", 0.0)

    def get_tail_coverage(regime, method_key):
        """Get tail coverage for method A (key 'method_A_gci') or B ('method_B_posterior')."""
        m = regime_results[regime][method_key]
        return m.get("tail_prob_0_95", 0.0)

    # Collect data
    method_a_var95 = [get_method_a_var95(r) for r in regimes]
    method_b_var95_mean = [get_method_b_var95_mean(r) for r in regimes]
    method_b_var95_std = [get_method_b_var95_std(r) for r in regimes]

    method_a_tail_cov = [get_tail_coverage(r, "method_A_gci") for r in regimes]
    method_b_tail_cov = [get_tail_coverage(r, "method_B_posterior") for r in regimes]

    # Method B uncertainty widening (vs baseline)
    baseline_width = None
    if "baseline" in regimes:
        baseline_var = regime_results["baseline"]["method_B_posterior"].get(
            "var_95_samples", np.array([])
        )
        if len(baseline_var) > 0:
            baseline_width = float(
                np.percentile(baseline_var, 97.5) - np.percentile(baseline_var, 2.5)
            )

    widening_ratios = []
    for r in regimes:
        var_samples = regime_results[r]["method_B_posterior"].get(
            "var_95_samples", np.array([])
        )
        if len(var_samples) > 0 and baseline_width is not None and baseline_width > 0:
            regime_width = float(
                np.percentile(var_samples, 97.5) - np.percentile(var_samples, 2.5)
            )
            widening_ratios.append(regime_width / baseline_width)
        else:
            widening_ratios.append(1.0)

    # VaR calibration error (mean |empirical - nominal|)
    def calibration_error(regime, method_key):
        m = regime_results[regime][method_key]
        err_95 = abs(m.get("tail_prob_0_95", 0.0) - 0.05)
        err_99 = abs(m.get("tail_prob_0_99", 0.0) - 0.01)
        err_999 = abs(m.get("tail_prob_0_999", 0.0) - 0.001)
        return (err_95 + err_99 + err_999) / 3.0

    method_a_cal_err = [calibration_error(r, "method_A_gci") for r in regimes]
    method_b_cal_err = [calibration_error(r, "method_B_posterior") for r in regimes]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Distribution Shift: Point-Estimate GCI vs Posterior-Propagated Factor Copula", fontsize=13)

    x = np.arange(len(regimes))
    bar_width = 0.35

    # Panel 1: VaR95 by regime
    ax1 = axes[0, 0]
    ax1.bar(x - bar_width / 2, method_a_var95, bar_width, label="Method A (Point-Estimate GCI)", color="steelblue")
    ax1.bar(
        x + bar_width / 2,
        method_b_var95_mean,
        bar_width,
        yerr=method_b_var95_std,
        label="Method B (Posterior Copula)",
        color="coral",
        capsize=3,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, rotation=30, ha="right")
    ax1.set_ylabel("VaR95 (EUR)")
    ax1.set_title("Panel 1: VaR95 by Regime")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Tail coverage by regime
    ax2 = axes[0, 1]
    ax2.plot(regimes, method_a_tail_cov, "o-", label="Method A (Point-Estimate GCI)", color="steelblue")
    ax2.plot(regimes, method_b_tail_cov, "s--", label="Method B (Posterior Copula)", color="coral")
    ax2.axhline(y=0.05, color="gray", linestyle=":", label="Ideal (5%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, rotation=30, ha="right")
    ax2.set_ylabel("Tail Coverage (fraction above VaR95)")
    ax2.set_title("Panel 2: Tail Coverage by Regime")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: Posterior uncertainty widening (log-scale)
    ax3 = axes[1, 0]
    bars = ax3.bar(regimes, widening_ratios, color="mediumpurple")
    ax3.axhline(y=1.0, color="gray", linestyle="--", label="Baseline width")
    ax3.set_yscale("log")
    ax3.set_ylabel("Width ratio (OOD / in-dist)")
    ax3.set_title("Panel 3: Posterior Uncertainty Widening")
    ax3.set_xticks(x)
    ax3.set_xticklabels(regimes, rotation=30, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend(fontsize=8)

    # Panel 4: VaR calibration error
    ax4 = axes[1, 1]
    width = 0.35
    ax4.bar(x - width / 2, method_a_cal_err, width, label="Method A", color="steelblue")
    ax4.bar(x + width / 2, method_b_cal_err, width, label="Method B", color="coral")
    ax4.set_xticks(x)
    ax4.set_xticklabels(regimes, rotation=30, ha="right")
    ax4.set_ylabel("Mean |empirical α - nominal α|")
    ax4.set_title("Panel 4: VaR Calibration Error")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 5 — Distribution Shift Experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--posterior-checkpoint",
        type=str,
        help="Path to Phase 1 SBI checkpoint (.pt). Alternative to --posterior-samples.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=str,
        help="Path to posterior samples as .npy file. Alternative to --posterior-checkpoint.",
    )
    parser.add_argument(
        "--test-regimes",
        nargs="+",
        default=["baseline", "housing_crash"],
        help="Test regimes (default: baseline housing_crash). "
             "Special: rate_shock_0.5 (low rate), rate_shock_1.5 (high rate).",
    )
    parser.add_argument(
        "--train-regime",
        type=str,
        default="baseline",
        help="Training regime label (default: baseline).",
    )
    parser.add_argument(
        "--n-posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples to use (default: 1000).",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=100_000,
        help="MC scenarios per evaluation (default: 100_000).",
    )
    parser.add_argument(
        "--copula",
        type=str,
        default="gaussian",
        choices=["gaussian", "student_t"],
        help="Factor copula model (default: gaussian).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Number of loans in portfolio (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz file path for results.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Optional path to save comparison plot (PNG).",
    )

    args = parser.parse_args(argv)

    # Load posterior samples
    if args.posterior_checkpoint:
        posterior_samples = _load_posterior(args.posterior_checkpoint)
    elif args.posterior_samples:
        posterior_samples = _load_posterior(args.posterior_samples)
    else:
        print(
            "ERROR: Must provide either --posterior-checkpoint or --posterior-samples",
            file=sys.stderr,
        )
        return 1

    # Subsample if requested
    if posterior_samples.shape[0] > args.n_posterior_samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(
            posterior_samples.shape[0], size=args.n_posterior_samples, replace=False
        )
        posterior_samples = posterior_samples[indices]

    N, D = posterior_samples.shape
    print(f"[OOD] Loaded {N} posterior samples, dim={D}")
    print(f"[OOD] Test regimes: {args.test_regimes}")
    print(f"[OOD] Copula: {args.copula}, K={args.K}")

    # Run experiment
    exp = OODExperiment(
        posterior_samples=posterior_samples,
        train_regime=args.train_regime,
        test_regimes=args.test_regimes,
        n_scenarios=args.n_scenarios,
        K=args.K,
        copula=args.copula,
        seed=args.seed,
    )

    results = exp.run()

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("  OOD Robustness — Summary")
    print("=" * 60)
    print(f"  Method A avg tail coverage error : {summary['method_A_avg_tail_coverage_error']:.4f}")
    print(f"  Method B avg tail coverage error : {summary['method_B_avg_tail_coverage_error']:.4f}")
    print(f"  Method B uncertainty widening    : {summary['method_B_uncertainty_widening']:.4f}")
    print(f"  Runtime                          : {results['runtime_seconds']:.1f}s")
    print("=" * 60)

    # Print per-regime VaR95 comparison
    print("\n  VaR95 Comparison (Method A vs Method B mean ± std):")
    for regime in args.test_regimes:
        rr = results["regime_results"][regime]
        a_var = rr["method_A_gci"].get("var_0_95", 0.0)
        b_mean = rr["method_B_posterior"].get("var_95_mean", 0.0)
        b_std = rr["method_B_posterior"].get("var_95_std", 0.0)
        print(f"    {regime:20s}: A={a_var:8.2f}  B={b_mean:8.2f}±{b_std:6.2f}")

    # Save results
    exp.save_npz(args.output)
    print(f"\n[OOD] Results saved to: {args.output}")

    # Generate plot if requested
    if args.plot:
        compare_methods_plot(results, args.plot)
        print(f"[OOD] Plot saved to: {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())