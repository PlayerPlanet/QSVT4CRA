"""
QSVT Degree Sweep — CLI entry point for Phase 4 experiments.

Runs QSVT circuits across multiple polynomial degrees and computes
error metrics vs classical ground truth.

Usage:
    python -m experiments.qsvt_sweep --degrees 16 32 64 128 256 512 1024 \
 --posterior-checkpoint posterior_samples.npz \
                                      --output results.npz

Architecture: D4 (docs/architecture.md §4)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def run_sweep(
    degrees: List[int],
    posterior_samples: Optional[np.ndarray] = None,
    K: int = 10,
    target_loss: float = 0.5,
    n_shots: int = 10000,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run QSVT degree sweep.

    Parameters
    ----------
    degrees : list of int
        Polynomial degrees to sweep.
    posterior_samples : np.ndarray, optional, shape (N, D)
        Posterior samples from SBI.
    K : int, default 10
        Number of loans (state qubits).
    target_loss : float, default 0.5
        Target loss threshold.
    n_shots : int, default 10000
        Number of measurement shots per circuit.
    output_path : str, optional
        Path to save results NPZ file.

    Returns
    -------
    results : dict
        Results dictionary with per-degree metrics.
    """
    from loader.posterior_factor_copula import PosteriorFactorCopulaLoader
    from qsvt.circuit import QSVTRiskCircuit
    from metrics.quantum_error import quantum_vs_classical_error
    from metrics.var_cvar import var_cvar

    # Default theta if no posterior provided
    if posterior_samples is None:
        rng = np.random.default_rng(42)
        factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
        p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
        tail_dep = np.array(0.0, dtype=np.float32)
        copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)
        theta = np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])
        posterior_samples = theta[None, :]

    # Use first posterior sample for sweep
    theta = posterior_samples[0]

    # LGD array
    lgd = np.full(K, 0.40, dtype=np.float32)

    # Classical ground truth
    from copula.gaussian import GaussianFactorCopula
    copula = GaussianFactorCopula(K=K, seed=42)
    _, losses = copula.sample(theta, n_samples=100000)
    classical_metrics = var_cvar(losses, alphas=[0.95, 0.99, 0.999])
    classical_var_95 = classical_metrics["var_0_95"]
    classical_cvar_95 = classical_metrics["cvar_0_95"]

    results = {
        "degrees": degrees,
        "K": K,
        "target_loss": target_loss,
        "n_shots": n_shots,
        "classical_var_95": classical_var_95,
        "classical_cvar_95": classical_cvar_95,
        "per_degree": {},
    }

    t0 = time.time()

    for degree in degrees:
        print(f"Processing degree {degree}...")

        try:
            # Build loader
            loader = PosteriorFactorCopulaLoader(
                theta=theta,
                K=K,
                max_loss=float(np.sum(lgd)),
                name=f"PFC_K{K}",
            )

            # Build QSVTRiskCircuit
            circuit = QSVTRiskCircuit(
                loader=loader,
                target_loss=target_loss,
                degree=degree,
                threshold=0.5,
                name=f"QSVTrisk_d{degree}",
            )

            # Compute error metrics
            error_result = quantum_vs_classical_error(
                circuit,
                classical_var_95 / np.sum(lgd),  # Normalize to [0, 1]
                n_shots=n_shots,
            )

            results["per_degree"][degree] = {
                "quantum_estimate": error_result["quantum_estimate"],
                "classical_value": error_result["classical_value"],
                "abs_error": error_result["abs_error"],
                "rel_error": error_result["rel_error"],
                "ci_95": error_result["ci_95"],
                "circuit_depth": circuit.depth(),
                "num_qubits": circuit.num_qubits,
            }

        except Exception as e:
            print(f"  Error at degree {degree}: {e}")
            results["per_degree"][degree] = {
                "error": str(e),
            }

    results["runtime_seconds"] = time.time() - t0

    # Save results
    if output_path:
        _save_results(results, output_path)

    return results


def _save_results(results: dict, output_path: str) -> None:
    """Save results to NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy-serializable format
    save_dict = {
        "degrees": np.array(results["degrees"]),
        "K": np.array(results["K"]),
        "target_loss": np.array(results["target_loss"]),
        "n_shots": np.array(results["n_shots"]),
        "classical_var_95": np.array(results["classical_var_95"]),
        "classical_cvar_95": np.array(results["classical_cvar_95"]),
        "runtime_seconds": np.array(results["runtime_seconds"]),
    }

    # Per-degree results
    for degree, metrics in results["per_degree"].items():
        prefix = f"d{degree}_"
        if "error" in metrics:
            save_dict[f"{prefix}error"] = metrics["error"]
        else:
            save_dict[f"{prefix}quantum_estimate"] = np.array(metrics["quantum_estimate"])
            save_dict[f"{prefix}classical_value"] = np.array(metrics["classical_value"])
            save_dict[f"{prefix}abs_error"] = np.array(metrics["abs_error"])
            save_dict[f"{prefix}rel_error"] = np.array(metrics["rel_error"])
            save_dict[f"{prefix}ci_95_lower"] = np.array(metrics["ci_95"][0])
            save_dict[f"{prefix}ci_95_upper"] = np.array(metrics["ci_95"][1])
            save_dict[f"{prefix}circuit_depth"] = np.array(metrics["circuit_depth"])
            save_dict[f"{prefix}num_qubits"] = np.array(metrics["num_qubits"])

    np.savez(output_path, **save_dict)
    print(f"Results saved to {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QSVT Degree Sweep Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024],
        help="Polynomial degrees to sweep",
    )
    parser.add_argument(
        "--posterior-checkpoint",
        type=str,
        default=None,
        help="Path to posterior samples NPZ file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/qsvt_sweep_results.npz",
        help="Output path for results",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Number of loans (state qubits)",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=0.5,
        help="Target loss threshold",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=10000,
        help="Number of measurement shots",
    )

    args = parser.parse_args()

    # Load posterior samples if provided
    posterior_samples = None
    if args.posterior_checkpoint:
        if str(args.posterior_checkpoint).endswith(".pt"):
            import torch
            ckpt = torch.load(args.posterior_checkpoint, map_location="cpu", weights_only=False)
            posterior_samples = (
                ckpt.get("posterior_samples")
                or ckpt.get("samples")
                or ckpt.get("theta")
                or ckpt.get("posterior")
            )
        else:
            data = np.load(args.posterior_checkpoint)
            if "theta_samples" in data:
                posterior_samples = data["theta_samples"]
            elif "posterior_samples" in data:
                posterior_samples = data["posterior_samples"]
            else:
                # Try to find any array with2D shape
                for key in data.files:
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[1] > 0:
                        posterior_samples = arr
                        break

    print(f"Running QSVT sweep with degrees: {args.degrees}")
    print(f"K={args.K}, target_loss={args.target_loss}, n_shots={args.n_shots}")

    results = run_sweep(
        degrees=args.degrees,
        posterior_samples=posterior_samples,
        K=args.K,
        target_loss=args.target_loss,
        n_shots=args.n_shots,
        output_path=args.output,
    )

    print(f"\nSweep complete in {results['runtime_seconds']:.1f}s")
    print(f"Classical VaR@95%: {results['classical_var_95']:.4f}")

    for degree in args.degrees:
        metrics = results["per_degree"].get(degree, {})
        if "error" in metrics:
            print(f"  Degree {degree}: ERROR - {metrics['error']}")
        else:
            print(
                f"  Degree {degree}: quantum={metrics['quantum_estimate']:.4f}, "
                f"abs_err={metrics['abs_error']:.4f}, rel_err={metrics['rel_error']:.4f}"
            )


if __name__ == "__main__":
    main()
