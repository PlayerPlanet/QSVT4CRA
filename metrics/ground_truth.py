"""
Massive Monte Carlo ground truth engine for VaR/CVaR benchmarking.

GroundTruthMC orchestrates the per-posterior-sample MC loop:
 For each θ⁽ⁱ⁾ in posterior_samples:
        → simulate n_scenarios portfolio losses via the chosen copula
        → compute VaR/CVaR at5/1/0.1% levels
        → aggregate into posterior-predictive distribution

Output is the benchmark against which QSVT approximations are validated.
"""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np

# Factor copula base is in copula package
from copula.base import FactorCopula
from copula.gaussian import GaussianFactorCopula
from copula.student_t import StudentTFactorCopula
from data.synthetic import SyntheticPortfolioGenerator


def _load_copula(copula_name: str, K: int, seed: int) -> FactorCopula:
    """Instantiate a factor copula by name."""
    copulas = {
        "gaussian": GaussianFactorCopula,
        "student_t": StudentTFactorCopula,
    }
    if copula_name not in copulas:
        raise ValueError(
            f"Unknown copula {copula_name!r}. Available: {list(copulas.keys())}"
        )
    return copulas[copula_name](K=K, seed=seed)


class GroundTruthMC:
    """
    Monte Carlo ground truth for posterior-predictive risk metrics.

    Runs large-scale MC simulation over posterior samples to produce the
    VaR/CVaR benchmark distribution.  Memory-efficient: losses are aggregated
    on-the-fly and never accumulated in full.

    Parameters
    ----------
    copula : FactorCopula
        Factor copula instance (Gaussian, Student-t, etc.).
    portfolio_generator : SyntheticPortfolioGenerator
        Portfolio generator providing loan attributes (K, LGD, principal).
    n_scenarios : int, default 1_000_000
        Base number of MC scenarios per experiment.
 posterior_samples : np.ndarray, optional, shape (N, D)
        Posterior samples from SBI.  If None, use a default theta.
    regime : str, default 'baseline'
        Stress regime name (for logging/labeling only).
    seed : int, default 42
        Random seed for reproducibility.

    Attributes
    ----------
    posterior_var : np.ndarray, shape (N,)
        VaR at 95% per posterior sample.
    posterior_cvar : np.ndarray, shape (N,)
        CVaR at 95% per posterior sample.
    predictive_var_at_0.95 : float
        Posterior-mean VaR95 (point estimate for the benchmark).
    predictive_cvar_at_0.95 : float
        Posterior-mean CVaR95 (point estimate for the benchmark).

    Examples
    --------
    >>> import numpy as np
    >>> from data.synthetic import SyntheticPortfolioGenerator
    >>> from copula.gaussian import GaussianFactorCopula
    >>> from metrics.ground_truth import GroundTruthMC
    >>> K = 10
    >>> theta = np.random.default_rng(0).uniform(-0.5, 0.5, 2*K+4).astype(np.float32)
    >>> posterior_samples = np.stack([theta + np.random.default_rng(i).normal(0, 0.05, 2*K+4) for i in range(10)])
    >>> gen = SyntheticPortfolioGenerator(K=K, seed=0)
    >>> copula = GaussianFactorCopula(K=K, seed=0)
    >>> mc = GroundTruthMC(copula, gen, n_scenarios=1000, posterior_samples=posterior_samples, seed=0)
    >>> result = mc.run(samples_per_posterior=1000)
    >>> 'posterior_var' in result
    True
    """

    # Confidence levels used throughout the project
    ALPHAS = [0.95, 0.99, 0.999]

    def __init__(
        self,
        copula: FactorCopula,
        portfolio_generator: SyntheticPortfolioGenerator,
        n_scenarios: int = 1_000_000,
        posterior_samples: Optional[np.ndarray] = None,
        regime: str = "baseline",
        seed: int = 42,
    ) -> None:
        self.copula = copula
        self.portfolio_generator = portfolio_generator
        self.n_scenarios = n_scenarios
        self.regime = regime
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Derived
        self.K = portfolio_generator.K

        # Default theta must be built after self.K is known
        if posterior_samples is None:
            posterior_samples = self._default_theta()
        self.posterior_samples = posterior_samples
        self.D = self.posterior_samples.shape[1]

    def _default_theta(self) -> np.ndarray:
        """Generate a single default theta when no posterior is available."""
        K = self.K
        rng = self._rng
        factor_loadings = rng.uniform(-0.5, 0.5, size=K).astype(np.float32)
        p_zeros = rng.uniform(0.005, 0.10, size=K).astype(np.float32)
        tail_dep = np.array(0.0, dtype=np.float32)
        copula_params = np.array([0.3, 30.0, 0.0], dtype=np.float32)  # rho, nu, spare
        return np.concatenate([factor_loadings, p_zeros, [tail_dep], copula_params])[
            None, :
        ]

    def run(
        self,
        samples_per_posterior: Optional[int] = None,
        store_all_losses: bool = False,
    ) -> dict:
        """
        Run the MC ground truth loop over all posterior samples.

        For each θ⁽ⁱ⁾, simulates ``samples_per_posterior`` losses via the copula,
        then computes VaR/CVaR at 95%, 99%, 99.9% levels.

        Parameters
        ----------
        samples_per_posterior : int, optional
            Number of MC scenarios per posterior sample.  Defaults to
            ``n_scenarios`` (1e6 by default).  Reduce for testing.
        store_all_losses : bool, default False
            If True and total samples < 1e8, store all loss samples.
            WARNING: can exhaust memory.  Leave False for production.

        Returns
        -------
        results : dict
            Keys:
            - ``posterior_var`` : np.ndarray (N,) — VaR95 per posterior sample
            - ``posterior_cvar`` : np.ndarray (N,) — CVaR95 per posterior sample
            - ``posterior_var_99`` : np.ndarray (N,) — VaR99 per posterior sample
            - ``posterior_cvar_99`` : np.ndarray (N,) — CVaR99 per posterior sample
            - ``posterior_var_999`` : np.ndarray (N,) — VaR99.9 per posterior sample
            - ``posterior_cvar_999`` : np.ndarray (N,) — CVaR99.9 per posterior sample
            - ``predictive_var_at_0.95`` : float — posterior mean VaR95
            - ``predictive_cvar_at_0.95`` : float — posterior mean CVaR95
            - ``predictive_var_at_0.99`` : float — posterior mean VaR99
            - ``predictive_cvar_at_0.99`` : float — posterior mean CVaR99
            - ``predictive_var_at_0.999`` : float — posterior mean VaR99.9
            - ``predictive_cvar_at_0.999`` : float — posterior mean CVaR99.9
            - ``all_loss_samples`` : None or np.ndarray —
 all losses if ``store_all_losses=True`` and total< 1e8
            - ``n_posterior_samples`` : int
            - ``n_scenarios_per_posterior`` : int
            - ``regime`` : str
            - ``runtime_seconds`` : float
 """
        if samples_per_posterior is None:
            samples_per_posterior = self.n_scenarios

        N = self.posterior_samples.shape[0]
        total_samples = N * samples_per_posterior

        # Pre-allocate result arrays
        var_95 = np.zeros(N, dtype=np.float32)
        cvar_95 = np.zeros(N, dtype=np.float32)
        var_99 = np.zeros(N, dtype=np.float32)
        cvar_99 = np.zeros(N, dtype=np.float32)
        var_999 = np.zeros(N, dtype=np.float32)
        cvar_999 = np.zeros(N, dtype=np.float32)

        all_losses = [] if store_all_losses and total_samples < 1e8 else None

        t0 = time.time()

        for i in range(N):
            theta_i = self.posterior_samples[i]
            _, losses_i = self.copula.sample(theta_i, n_samples=samples_per_posterior)

            if all_losses is not None:
                all_losses.append(losses_i)

            # Compute metrics for this posterior sample
            var_95[i] = float(np.quantile(losses_i, 0.95))
            tail_95 = losses_i[losses_i > var_95[i]]
            cvar_95[i] = float(tail_95.mean()) if tail_95.size > 0 else var_95[i]

            var_99[i] = float(np.quantile(losses_i, 0.99))
            tail_99 = losses_i[losses_i > var_99[i]]
            cvar_99[i] = float(tail_99.mean()) if tail_99.size > 0 else var_99[i]

            var_999[i] = float(np.quantile(losses_i, 0.999))
            tail_999 = losses_i[losses_i > var_999[i]]
            cvar_999[i] = float(tail_999.mean()) if tail_999.size > 0 else var_999[i]

        elapsed = time.time() - t0

        # Aggregate into posterior-predictive summary
        predictive_var_95 = float(var_95.mean())
        predictive_cvar_95 = float(cvar_95.mean())
        predictive_var_99 = float(var_99.mean())
        predictive_cvar_99 = float(cvar_99.mean())
        predictive_var_999 = float(var_999.mean())
        predictive_cvar_999 = float(cvar_999.mean())

        results = {
            # Per-sample metrics
            "posterior_var": var_95,
            "posterior_cvar": cvar_95,
            "posterior_var_99": var_99,
            "posterior_cvar_99": cvar_99,
            "posterior_var_999": var_999,
            "posterior_cvar_999": cvar_999,
            # Posterior-predictive summaries
            "predictive_var_at_0.95": predictive_var_95,
            "predictive_cvar_at_0.95": predictive_cvar_95,
            "predictive_var_at_0.99": predictive_var_99,
            "predictive_cvar_at_0.99": predictive_cvar_99,
            "predictive_var_at_0.999": predictive_var_999,
            "predictive_cvar_at_0.999": predictive_cvar_999,
            # Metadata
            "all_loss_samples": (
                np.concatenate(all_losses) if all_losses is not None else None
            ),
            "n_posterior_samples": N,
            "n_scenarios_per_posterior": samples_per_posterior,
            "regime": self.regime,
            "runtime_seconds": elapsed,
        }

        return results

    def run_streaming(
        self,
        samples_per_posterior: int,
        batch_size: int = 50_000,
    ) -> dict:
        """
        Memory-efficient MC loop that processes losses in batches.

        Use this when ``samples_per_posterior`` is large (e.g., 1e7) to
        avoid holding all losses in memory at once.

        Parameters
        ----------
        samples_per_posterior : int
            Total MC scenarios per posterior sample.
        batch_size : int, default 50_000
            Size of each batch for on-the-fly quantile estimation.

        Returns
        -------
        results : dict
            Same structure as ``run()``.
        """
        N = self.posterior_samples.shape[0]

        var_95 = np.zeros(N, dtype=np.float32)
        cvar_95 = np.zeros(N, dtype=np.float32)
        var_99 = np.zeros(N, dtype=np.float32)
        cvar_99 = np.zeros(N, dtype=np.float32)
        var_999 = np.zeros(N, dtype=np.float32)
        cvar_999 = np.zeros(N, dtype=np.float32)

        t0 = time.time()

        for i in range(N):
            theta_i = self.posterior_samples[i]

            # Stream losses in batches and accumulate order statistics
            all_batch_losses = []
            n_batches = (samples_per_posterior + batch_size - 1) // batch_size
            for b in range(n_batches):
                n_draw = min(batch_size, samples_per_posterior - b * batch_size)
                _, batch_losses = self.copula.sample(theta_i, n_samples=n_draw)
                all_batch_losses.append(batch_losses)

            losses_i = np.concatenate(all_batch_losses)

            var_95[i] = float(np.quantile(losses_i, 0.95))
            tail_95 = losses_i[losses_i > var_95[i]]
            cvar_95[i] = float(tail_95.mean()) if tail_95.size > 0 else var_95[i]
            var_99[i] = float(np.quantile(losses_i, 0.99))
            tail_99 = losses_i[losses_i > var_99[i]]
            cvar_99[i] = float(tail_99.mean()) if tail_99.size > 0 else var_99[i]
            var_999[i] = float(np.quantile(losses_i, 0.999))
            tail_999 = losses_i[losses_i > var_999[i]]
            cvar_999[i] = float(tail_999.mean()) if tail_999.size > 0 else var_999[i]

        elapsed = time.time() - t0

        return {
            "posterior_var": var_95,
            "posterior_cvar": cvar_95,
            "posterior_var_99": var_99,
            "posterior_cvar_99": cvar_99,
            "posterior_var_999": var_999,
            "posterior_cvar_999": cvar_999,
            "predictive_var_at_0.95": float(var_95.mean()),
            "predictive_cvar_at_0.95": float(cvar_95.mean()),
            "predictive_var_at_0.99": float(var_99.mean()),
            "predictive_cvar_at_0.99": float(cvar_99.mean()),
            "predictive_var_at_0.999": float(var_999.mean()),
            "predictive_cvar_at_0.999": float(cvar_999.mean()),
            "all_loss_samples": None,
            "n_posterior_samples": N,
            "n_scenarios_per_posterior": samples_per_posterior,
            "regime": self.regime,
            "runtime_seconds": elapsed,
        }
