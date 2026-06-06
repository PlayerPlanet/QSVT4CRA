#!/usr/bin/env python
"""
MC Ground Truth CLI — Phase 3 of QSVT4CRA Research Run.

Massive Monte Carlo simulation to establish VaR/CVaR benchmark against
which QSVT approximations will be validated.

Usage
------
# Run with a posterior checkpoint from Phase 1:
python -m experiments.mc_ground_truth \\
    --posterior-checkpoint sbi_pipeline/checkpoint_npe.pt \\
    --n-scenarios 1000000 \\
    --copula gaussian \\
    --regime baseline \\
    --output results/ground_truth_gaussian_baseline.npz

# Run with direct posterior samples file:
python -m experiments.mc_ground_truth \\
    --posterior-samples posterior_samples.npy \\
    --n-scenarios 1000000 \\
    --copula student_t \\
    --regime housing_crash \\
    --output results/ground_truth_t_housing_crash.npz

# Quick test (small N, small scenarios):
python -m experiments.mc_ground_truth \\
    --posterior-samples posterior_samples.npy \\
    --n-scenarios 1000 \\
    --copula gaussian \\
    --regime baseline \\
    --output results/test_ground_truth.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from copula.gaussian import GaussianFactorCopula
from copula.student_t import StudentTFactorCopula
from data.synthetic import SyntheticPortfolioGenerator
from metrics.ground_truth import GroundTruthMC


def _load_posterior(path: str) -> np.ndarray:
    """
    Load posterior samples from a Phase 1 checkpoint or npy file.

    Supported formats:
    - ``.npy`` / ``.npz`` : direct numpy array
    - ``.pt`` (PyTorch) : Phase 1 SBI checkpoint dict with key 'posterior'
 or 'samples'

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    Returns
    -------
    samples : np.ndarray, shape (N, D)
        Posterior samples.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:
            # Single row → reshape to (1, D)
            arr = arr[None, :]
        return arr.astype(np.float32)

    if suffix == ".npz":
        with np.load(path) as f:
            # Try known keys
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

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                for key in ("posterior", "samples", "theta"):
                    if key in ckpt:
                        arr = ckpt[key]
                        if isinstance(arr, np.ndarray):
                            if arr.ndim == 1:
                                arr = arr[None, :]
                            return arr.astype(np.float32)
                        # PyTorch tensor
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Monte Carlo ground truth for VaR/CVaR benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--posterior-checkpoint",
        type=str,
        help=(
            "Path to Phase 1 SBI checkpoint (.pt). "
            "Used when --posterior-samples is not provided."
        ),
    )
    parser.add_argument(
        "--posterior-samples",
        type=str,
        help=(
            "Path to posterior samples as .npy file.  "
            "Alternative to --posterior-checkpoint."
        ),
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=1_000_000,
        help="Number of MC scenarios per posterior sample (default: 1_000_000)",
    )
    parser.add_argument(
        "--copula",
        type=str,
        default="gaussian",
        choices=["gaussian", "student_t"],
        help="Factor copula model (default: gaussian)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="baseline",
        choices=["baseline", "housing_crash", "rate_shock", "unemployment", "liquidity"],
        help="Stress regime (default: baseline)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz file path for results",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Number of loans in portfolio (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use memory-efficient streaming mode (batches of 50k). "
            "Recommended for n_scenarios >= 1e7."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Batch size for streaming mode (default: 50_000)",
    )

    args = parser.parse_args(argv)

    # --- Load posterior samples ---
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

    N, D = posterior_samples.shape
    print(f"[MC Ground Truth] Loaded {N} posterior samples, dim={D}")
    print(f"[MC Ground Truth] Copula: {args.copula}, Regime: {args.regime}")
    print(f"[MC Ground Truth] Scenarios per posterior: {args.n_scenarios:,}")

    # --- Instantiate copula and portfolio generator ---
    K = args.K
    copula = _get_copula(args.copula, K=K, seed=args.seed)
    portfolio_gen = SyntheticPortfolioGenerator(K=K, seed=args.seed)

    # --- Run MC ground truth ---
    mc = GroundTruthMC(
        copula=copula,
        portfolio_generator=portfolio_gen,
        n_scenarios=args.n_scenarios,
        posterior_samples=posterior_samples,
        regime=args.regime,
        seed=args.seed,
    )

    t0 = time.time()
    if args.streaming:
        print("[MC Ground Truth] Running in STREAMING mode...")
        results = mc.run_streaming(
            samples_per_posterior=args.n_scenarios,
            batch_size=args.batch_size,
        )
    else:
        results = mc.run(samples_per_posterior=args.n_scenarios)
    elapsed = time.time() - t0

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  MC Ground Truth — Summary")
    print("=" * 60)
    print(f"  Posterior samples : {results['n_posterior_samples']}")
    print(f"  Scenarios/sample   : {results['n_scenarios_per_posterior']:,}")
    print(f"  Regime             : {results['regime']}")
    print(f"  Runtime : {elapsed:.1f}s")
    print()
    print(f"  Predictive VaR95   : EUR {results['predictive_var_at_0.95']:,.2f}")
    print(f"  Predictive CVaR95  : EUR {results['predictive_cvar_at_0.95']:,.2f}")
    print(f"  Predictive VaR99   : EUR {results['predictive_var_at_0.99']:,.2f}")
    print(f"  Predictive CVaR99  : EUR {results['predictive_cvar_at_0.99']:,.2f}")
    print(f"  Predictive VaR99.9 : EUR {results['predictive_var_at_0.999']:,.2f}")
    print(f"  Predictive CVaR99.9: EUR {results['predictive_cvar_at_0.999']:,.2f}")
    print("=" * 60)

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only save what's needed (not all loss samples)
    save_dict = {
        "posterior_var": results["posterior_var"],
        "posterior_cvar": results["posterior_cvar"],
        "posterior_var_99": results["posterior_var_99"],
        "posterior_cvar_99": results["posterior_cvar_99"],
        "posterior_var_999": results["posterior_var_999"],
        "posterior_cvar_999": results["posterior_cvar_999"],
        "predictive_var_at_0.95": results["predictive_var_at_0.95"],
        "predictive_cvar_at_0.95": results["predictive_cvar_at_0.95"],
        "predictive_var_at_0.99": results["predictive_var_at_0.99"],
        "predictive_cvar_at_0.99": results["predictive_cvar_at_0.99"],
        "predictive_var_at_0.999": results["predictive_var_at_0.999"],
        "predictive_cvar_at_0.999": results["predictive_cvar_at_0.999"],
        "n_posterior_samples": results["n_posterior_samples"],
        "n_scenarios_per_posterior": results["n_scenarios_per_posterior"],
        "regime": results["regime"],
        "runtime_seconds": results["runtime_seconds"],
        "copula": args.copula,
    }

    np.savez(output_path, **save_dict)
    print(f"\n[MC Ground Truth] Results saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
