#!/usr/bin/env python
"""
SBI Training CLI — Phase 1 of QSVT4CRA Research Run.

Trains a Simulation-Based Inference posterior over factor-copula parameters
from observed apartment loan portfolio loss data. Three estimators supported:
  - NPE  (Neural Posterior Estimation, default; sequential proposal adaptation)
  - NLE  (Neural Likelihood Estimation; robustness check)
  - FM   (Conditional Masked Autoregressive Flow; flow-matching style training)

Usage
-----
# Quick test (NPE, 100 sims, 2 rounds, CPU):
python -m experiments.sbi_train \\
    --method npe \\
    --n-simulations 100 \\
    --n-rounds 2 \\
    --device cpu \\
    --output checkpoints/sbi_npe_smoke.pt

# Full NPE run (1000 sims × 10 rounds, GPU):
python -m experiments.sbi_train \\
    --method npe \\
    --n-simulations 1000 \\
    --n-rounds 10 \\
    --device cuda \\
    --K 10 \\
    --regime baseline \\
    --output checkpoints/sbi_npe_baseline.pt

# NLE robustness check:
python -m experiments.sbi_train \\
    --method nle \\
    --n-simulations 1000 \\
    --n-rounds 10 \\
    --device cuda \\
    --output checkpoints/sbi_nle_baseline.pt

# Flow matching (cMAF):
python -m experiments.sbi_train \\
    --method fm \\
    --n-simulations 1000 \\
    --n-rounds 10 \\
    --device cuda \\
    --output checkpoints/sbi_fm_baseline.pt

Notes
-----
- This is a STANDALONE training entry point. It does NOT call Slurm.
  See lumi_deployment/slurm_qsvt4cra_research.sh for the Slurm wrapper.
- The simulator is created on-the-fly using NumPyForwardSimulator.
- For multi-GPU training, see scale_runner.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.synthetic import SyntheticPortfolioGenerator
from data.stress_regimes import StressRegimeGenerator
from simulator.forward import NumPyForwardSimulator
from sbi_pipeline.posterior import (
    NPETrainingPipeline,
    NLETrainingPipeline,
    FlowMatchingTrainingPipeline,
)
from sbi_pipeline.utils import SBITrainingConfig, get_prior_from_bounds, get_logging_hook


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def _build_prior(K: int, regime: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct prior lower/upper bounds for factor-copula parameters.

    Parameters
    ----------
    K : int
        Number of loans in portfolio.
    regime : str
        Stress regime (affects prior width).

    Returns
    -------
    (low, high) : tuple of np.ndarray
        Lower and upper bounds, shape (2K + 4,).
        Index layout:
            [0:K]       factor_loadings (b_i)
            [K:2K]      default probabilities p_zeros
            [2K]        tail dependence
            [2K+1:2K+4] copula params (rho, nu, rotation)
    """
    # Factor loadings: bounded in [-0.7, 0.7]
    low_factor = np.full(K, -0.7, dtype=np.float32)
    high_factor = np.full(K, 0.7, dtype=np.float32)

    # Default probabilities: regime-dependent
    # Under stress, allow higher default rates
    p_max = 0.30 if regime in ("baseline",) else 0.50
    low_p = np.full(K, 0.001, dtype=np.float32)
    high_p = np.full(K, p_max, dtype=np.float32)

    # Tail dependence [0, 1]
    low_tail = np.array([0.0], dtype=np.float32)
    high_tail = np.array([1.0], dtype=np.float32)

    # Copula params (rho, nu, rotation)
    low_copula = np.array([0.0, 4.0, -1.0], dtype=np.float32)
    high_copula = np.array([0.95, 100.0, 1.0], dtype=np.float32)

    low = np.concatenate([low_factor, low_p, low_tail, low_copula])
    high = np.concatenate([high_factor, high_p, high_tail, high_copula])
    return low, high


def _make_simulator(K: int, regime: str, seed: int) -> NumPyForwardSimulator:
    """
    Build a NumPy forward simulator for the given K and regime.

    Parameters
    ----------
    K : int
        Number of loans.
    regime : str
        Stress regime name.
    seed : int
        Random seed.

    Returns
    -------
    NumPyForwardSimulator
    """
    portfolio_gen = SyntheticPortfolioGenerator(K=K, seed=seed)
    return NumPyForwardSimulator(portfolio_generator=portfolio_gen, regime=regime, seed=seed)


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

def _simulate_training_pairs(
    simulator: NumPyForwardSimulator,
    low: np.ndarray,
    high: np.ndarray,
    n: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Simulate (theta, x) training pairs by sampling theta from a uniform
    hyperrectangle and running the forward simulator.

    Parameters
    ----------
    simulator : NumPyForwardSimulator
        Forward simulator.
    low, high : np.ndarray
        Prior lower/upper bounds, shape (D,).
    n : int
        Number of training pairs to generate.
    seed : int
        Random seed.

    Returns
    -------
    list of (theta, x) tuples
    """
    rng = np.random.default_rng(seed)
    D = len(low)
    thetas = rng.uniform(low, high, size=(n, D)).astype(np.float32)
    xs = simulator.simulate(thetas, n_scenarios=1)
    return list(zip(thetas, xs))


# ---------------------------------------------------------------------------
# Training entry per method
# ---------------------------------------------------------------------------

def _train_npe(
    simulator: NumPyForwardSimulator,
    low: np.ndarray,
    high: np.ndarray,
    n_simulations: int,
    n_rounds: int,
    device: str,
    seed: int,
    wandb_project: str | None,
) -> "TrainingResult":
    """Train NPE pipeline."""
    prior = get_prior_from_bounds(low, high)
    pipeline = NPETrainingPipeline(
        prior=prior,
        hidden_features=50,
        num_transforms=4,
        device=device,
        seed=seed,
        wandb_project=wandb_project,
    )
    training_pairs = _simulate_training_pairs(
        simulator=simulator, low=low, high=high, n=n_simulations, seed=seed
    )
    return pipeline.train(
        training_pairs=training_pairs,
        n_rounds=n_rounds,
        n_simulations_per_round=max(50, n_simulations // n_rounds),
        batch_size=min(100, n_simulations),
        learning_rate=5e-4,
    )


def _train_nle(
    simulator: NumPyForwardSimulator,
    low: np.ndarray,
    high: np.ndarray,
    n_simulations: int,
    n_rounds: int,
    device: str,
    seed: int,
    wandb_project: str | None,
) -> "TrainingResult":
    """Train NLE pipeline."""
    prior = get_prior_from_bounds(low, high)
    pipeline = NLETrainingPipeline(
        prior=prior,
        hidden_features=50,
        num_transforms=4,
        device=device,
        seed=seed,
        wandb_project=wandb_project,
    )
    training_pairs = _simulate_training_pairs(
        simulator=simulator, low=low, high=high, n=n_simulations, seed=seed
    )
    return pipeline.train(
        training_pairs=training_pairs,
        n_rounds=n_rounds,
        n_simulations_per_round=max(50, n_simulations // n_rounds),
        batch_size=min(100, n_simulations),
        learning_rate=5e-4,
    )


def _train_fm(
    simulator: NumPyForwardSimulator,
    low: np.ndarray,
    high: np.ndarray,
    n_simulations: int,
    n_rounds: int,
    device: str,
    seed: int,
    wandb_project: str | None,
) -> "TrainingResult":
    """Train Flow Matching (cMAF) pipeline."""
    prior = get_prior_from_bounds(low, high)
    pipeline = FlowMatchingTrainingPipeline(
        prior=prior,
        hidden_features=50,
        num_transforms=4,
        device=device,
        seed=seed,
        wandb_project=wandb_project,
    )
    training_pairs = _simulate_training_pairs(
        simulator=simulator, low=low, high=high, n=n_simulations, seed=seed
    )
    return pipeline.train(
        training_pairs=training_pairs,
        n_rounds=n_rounds,
        n_simulations_per_round=max(50, n_simulations // n_rounds),
        batch_size=min(100, n_simulations),
        learning_rate=5e-4,
    )


# ---------------------------------------------------------------------------
# Checkpoint serialization
# ---------------------------------------------------------------------------

def _save_checkpoint(
    result,
    method: str,
    K: int,
    regime: str,
    n_simulations: int,
    n_rounds: int,
    seed: int,
    output: str,
) -> None:
    """
    Save training result to a torch checkpoint.

    The saved dict contains:
        - method : str
        - K, regime, n_simulations, n_rounds, seed
        - posterior_samples : np.ndarray, shape (N, D)
            Posterior samples drawn from the trained posterior.
        - log_probs, ess_values : training diagnostics
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw samples from the trained posterior
    try:
        n_samples = 1000
        samples = result.posterior.sample((n_samples,))
        if isinstance(samples, torch.Tensor):
            samples_np = samples.detach().cpu().numpy().astype(np.float32)
        else:
            samples_np = np.asarray(samples, dtype=np.float32)
    except Exception as e:
        print(f"WARNING: Could not draw samples from posterior: {e}", file=sys.stderr)
        samples_np = np.zeros((0, 2 * K + 4), dtype=np.float32)

    save_dict = {
        "method": method,
        "K": K,
        "regime": regime,
        "n_simulations": n_simulations,
        "n_rounds": n_rounds,
        "seed": seed,
        "posterior_samples": samples_np,
        "log_probs": result.log_probs,
        "ess_values": result.ess_values,
        "wandb_url": result.wandb_url,
    }
    torch.save(save_dict, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[SBI Train] Checkpoint saved: {output_path} ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for SBI training.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description="Train an SBI posterior over factor-copula parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["npe", "nle", "fm"],
        default="npe",
        help="SBI method: npe (default), nle, or fm (flow matching / cMAF).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Number of loans in portfolio (default 10).",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "housing_crash",
            "rate_shock_0.5",
            "rate_shock_1.5",
            "unemployment",
            "liquidity",
        ],
        help="Stress regime.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=1000,
        help="Number of (theta, x) training pairs to simulate (default 1000).",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=10,
        help="Number of NPE/NLE proposal-adaptation rounds (default 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cuda or cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional W&B project name. If None, no W&B logging.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/sbi_npe.pt",
        help="Output checkpoint path (.pt file).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional time limit in seconds; warns if exceeded (not enforced).",
    )

    args = parser.parse_args(argv)

    print("=" * 60)
    print("QSVT4CRA - Phase 1: SBI Posterior Training")
    print("=" * 60)
    print(f"  Method            : {args.method.upper()}")
    print(f"  K (loans)         : {args.K}")
    print(f"  Regime            : {args.regime}")
    print(f"  N simulations     : {args.n_simulations}")
    print(f"  N rounds          : {args.n_rounds}")
    print(f"  Device            : {args.device}")
    print(f"  Seed              : {args.seed}")
    print(f"  W&B project       : {args.wandb_project or '(none)'}")
    print(f"  Output            : {args.output}")
    print()

    # Force CPU if CUDA not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device=cuda but CUDA not available; falling back to CPU.")
        args.device = "cpu"

    # Build simulator and prior
    simulator = _make_simulator(K=args.K, regime=args.regime, seed=args.seed)
    low, high = _build_prior(K=args.K, regime=args.regime)
    print(f"[SBI Train] Prior dim: {len(low)} (= 2K+4 with K={args.K})")

    # Dispatch to method-specific trainer
    train_fn = {
        "npe": _train_npe,
        "nle": _train_nle,
        "fm": _train_fm,
    }[args.method]

    t0 = time.time()
    try:
        result = train_fn(
            simulator=simulator,
            low=low,
            high=high,
            n_simulations=args.n_simulations,
            n_rounds=args.n_rounds,
            device=args.device,
            seed=args.seed,
            wandb_project=args.wandb_project,
        )
    except Exception as e:
        print(f"ERROR: Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0

    # Print diagnostics
    print()
    print("=" * 60)
    print("  SBI Training — Summary")
    print("=" * 60)
    print(f"  Method            : {args.method.upper()}")
    print(f"  Rounds completed  : {len(result.log_probs)}")
    if result.log_probs:
        print(f"  Final log_prob    : {result.log_probs[-1]:.4f}")
        print(f"  Mean log_prob     : {np.mean(result.log_probs):.4f}")
    if result.ess_values:
        print(f"  Final ESS         : {result.ess_values[-1]:.1f}")
    print(f"  Wall time         : {elapsed:.1f}s")
    if result.wandb_url:
        print(f"  W&B run           : {result.wandb_url}")
    print("=" * 60)

    if args.time_limit and elapsed > args.time_limit:
        print(
            f"WARNING: Training took {elapsed:.1f}s, "
            f"exceeds --time-limit={args.time_limit}s",
            file=sys.stderr,
        )

    # Save checkpoint
    _save_checkpoint(
        result=result,
        method=args.method,
        K=args.K,
        regime=args.regime,
        n_simulations=args.n_simulations,
        n_rounds=args.n_rounds,
        seed=args.seed,
        output=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
