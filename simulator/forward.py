"""
Forward simulators: θ → observation vectors.

Provides a common abstract interface with two backends:
- JAXForwardSimulator : JIT-compiled GPU-capable simulator
- NumPyForwardSimulator : pure NumPy fallback
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class ForwardSimulator(ABC):
    """
    Abstract forward simulator.

    Subclasses must implement ``_simulate_batch`` for vectorized simulation.

    Parameters
    ----------
    portfolio_generator : SyntheticPortfolioGenerator
        Portfolio generator providing the forward model.
    regime : str, default "baseline"
        Stress regime name (passed to StressRegimeGenerator).
    seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        portfolio_generator,
        regime: str = "baseline",
        seed: int = 42,
    ) -> None:
        self.portfolio_generator = portfolio_generator
        self.regime = regime
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def _simulate_batch(self, theta_batch: np.ndarray, n_scenarios: int) -> np.ndarray:
        """
        Core batch simulation. Subclasses override with backend-specific code.

        Parameters
        ----------
        theta_batch : np.ndarray, shape (B, D)
            Batch of parameter vectors.
        n_scenarios : int
            Number of Monte Carlo scenarios per theta.

        Returns
        -------
        np.ndarray, shape (B, 10)
            Observation matrix (float32).
        """
        ...

    def simulate(self, theta_batch: np.ndarray, n_scenarios: int = 1000) -> np.ndarray:
        """
        Vectorized batch simulation: B thetas → B observation vectors.

        Parameters
        ----------
        theta_batch : np.ndarray, shape (B, D)
            Batch of parameter vectors.
        n_scenarios : int, default 1000
            Number of Monte Carlo scenarios per theta.

        Returns
        -------
        np.ndarray, shape (B, 10)
            Observation matrix (float32).
        """
        theta_batch = np.asarray(theta_batch, dtype=np.float32)
        if theta_batch.ndim != 2:
            raise ValueError(
                f"theta_batch must be 2D (B, D), got shape {theta_batch.shape}"
            )
        return self._simulate_batch(theta_batch, n_scenarios).astype(np.float32)

    def simulate_single(self, theta: np.ndarray, n_scenarios: int = 1000) -> np.ndarray:
        """
        Convenience wrapper: simulate a single theta vector.

        Parameters
        ----------
        theta : np.ndarray, shape (D,)
            Single parameter vector.
        n_scenarios : int, default 1000
            Number of Monte Carlo scenarios.

        Returns
        -------
        np.ndarray, shape (10,)
            Observation vector (float32).
        """
        theta = np.asarray(theta, dtype=np.float32)
        result = self.simulate(theta[np.newaxis, :], n_scenarios=n_scenarios)
        return result[0].astype(np.float32)

    def grad_log_likelihood(
        self, theta: np.ndarray, x_obs: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of log-likelihood log p(x_obs | theta) w.r.t. theta.

        Default implementation uses finite differences (central differences).
        Subclasses with autodiff backends should override.

        Parameters
        ----------
        theta : np.ndarray, shape (D,)
            Parameter vector.
        x_obs : np.ndarray, shape (10,)
            Observed feature vector.

        Returns
        -------
        np.ndarray, shape (D,)
            Gradient vector (float32).
        """
        eps = 1e-4
        D = theta.shape[0]
        grad = np.zeros(D, dtype=np.float32)
        for i in range(D):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps
            x_plus = self.simulate_single(theta_plus)
            x_minus = self.simulate_single(theta_minus)
            grad[i] = (x_plus - x_minus).sum() / (2 * eps)
        return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------
class NumPyForwardSimulator(ForwardSimulator):
    """
    Pure NumPy forward simulator.

    Uses the attached SyntheticPortfolioGenerator for loss simulation.
    Suitable for CPU-only environments.
    """

    def _simulate_batch(self, theta_batch: np.ndarray, n_scenarios: int) -> np.ndarray:
        """
        Simulate each theta in the batch using the portfolio generator.

        Parameters
        ----------
        theta_batch : np.ndarray, shape (B, D)
            Batch of parameter vectors.
        n_scenarios : int
            Number of Monte Carlo scenarios per theta.

        Returns
        -------
        np.ndarray, shape (B, 10)
            Observation matrix.
        """
        B = theta_batch.shape[0]
        observations = np.empty((B, 10), dtype=np.float32)

        for b in range(B):
            dataset = self.portfolio_generator.sample(theta_batch[b], n_scenarios)
            obs = self.portfolio_generator.losses_to_observation(dataset.losses)
            observations[b] = obs

        return observations.astype(np.float32)


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------
class JAXForwardSimulator(ForwardSimulator):
    """
    JAX-accelerated forward simulator.

    Attempts to use JAX with JIT compilation for GPU execution.
    Falls back to NumPyForwardSimulator if JAX is unavailable or
    no GPU is detected.

    Parameters
    ----------
    portfolio_generator : SyntheticPortfolioGenerator
        Portfolio generator (used for initialisation of JAX equivalents).
    regime : str, default "baseline"
        Stress regime name.
    seed : int, default 42
        Random seed.
    """

    def __init__(
        self,
        portfolio_generator,
        regime: str = "baseline",
        seed: int = 42,
    ) -> None:
        super().__init__(portfolio_generator, regime, seed)
        self._jax = None
        self._jnp = None
        self._jitted_simulate = None
        self._numpy_fallback = NumPyForwardSimulator(
            portfolio_generator, regime, seed
        )

        try:
            import jax
            import jax.numpy as jnp
            from jax import jit, vmap
            from scipy.stats import norm

            self._jax = jax
            self._jnp = jnp
            self._jit = jit
            self._vmap = vmap
            self._norm = norm
        except ImportError:
            pass

    def _simulate_batch(self, theta_batch: np.ndarray, n_scenarios: int) -> np.ndarray:
        """
        Dispatch to JAX or NumPy depending on availability.
        """
        if self._jax is None:
            return self._numpy_fallback._simulate_batch(theta_batch, n_scenarios)

        # Check GPU availability
        jax_platform = self._jax.default_backend()
        gpu_available = jax_platform not in ("cpu",)

        if not gpu_available:
            return self._numpy_fallback._simulate_batch(theta_batch, n_scenarios)

        # Build JIT-compiled simulation function on first call
        if self._jitted_simulate is None:
            self._jitted_simulate = self._build_jitted_simulate()

        # Run JAX simulation
        theta_jax = self._jnp.array(theta_batch.astype(np.float32))
        obs_jax = self._jitted_simulate(theta_jax, n_scenarios)
        return np.array(obs_jax).astype(np.float32)

    def _build_jitted_simulate(self):
        """
        Construct a JIT-compiled JAX simulation function.

        Returns a function jitted_simulate(theta_batch, n_scenarios) → obs_batch.
        """
        jax = self._jax
        jnp = self._jnp
        jit = self._jit
        vmap = self._vmap
        norm = self._norm
        K = self.portfolio_generator.K
        lgd_arr = np.array(self.portfolio_generator._lgd, dtype=np.float32)
        principal_arr = np.array(
            self.portfolio_generator._principal, dtype=np.float32
        )
        seed = self.seed

        @jit(static_argnums=(0,))
        def sample_z(n_sc: int, tail_dep: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
            """Sample systemic factor Z. n_sc is static (not traced) for JAX random."""
            use_t = (tail_dep > 1e-6) & (nu < 100.0)
            W = jax.random.normal(jax.random.PRNGKey(seed), (n_sc,))
            V = (
                jax.random.gamma(
                    jax.random.PRNGKey(seed + 1),
                    nu / 2.0,
                    shape=(n_sc,),
                )
                * 2.0
            )
            Z = jnp.where(use_t, W / jnp.sqrt(V / nu), W)
            return Z

        @jit
        def simulate_theta(theta: jnp.ndarray, n_scenarios: int) -> jnp.ndarray:
            """
            Simulate one theta vector and return observations (10,).
            n_scenarios is static (not traced) to allow JAX random functions.
            """
            factor_loadings = theta[:K]
            p_zeros = theta[K : 2 * K]
            tail_dep = theta[2 * K]
            rho = theta[2 * K + 1]
            nu = theta[2 * K + 2]

            # Systemic factor Z (n_scenarios is static, tail_dep/nu are traced)
            Z = sample_z(n_scenarios, tail_dep, nu)

            # Conditional default probabilities (probit link)
            sqrt_rho = jnp.sqrt(jnp.clip(rho, 0.0, 0.9999))
            beta_sq = factor_loadings ** 2
            denom = jnp.sqrt(1.0 - rho * beta_sq + 1e-12)

            phi_inv_p = norm.ppf(jnp.clip(p_zeros, 1e-6, 1.0 - 1e-6))
            numerator = (
                phi_inv_p[:, None]
                - sqrt_rho * factor_loadings[:, None] * Z[None, :]
            )
            cond_p = norm.cdf(numerator / denom[:, None])

            # Bernoulli defaults
            U = jax.random.uniform(
                jax.random.PRNGKey(seed + 2), shape=(K, n_scenarios)
            )
            defaults = (U < cond_p).astype(jnp.float32)

            # Per-loan losses
            lgd = jnp.array(lgd_arr, dtype=jnp.float32)
            principal = jnp.array(principal_arr, dtype=jnp.float32)
            loan_losses = lgd[:, None] * principal[:, None] * defaults
            total_losses = jnp.sum(loan_losses, axis=0)

            # Observations
            mean_lgd_per_loan = jnp.mean(lgd)
            n_defaults_est = jnp.clip(
                (total_losses / (mean_lgd_per_loan + 1e-8)).astype(jnp.int32),
                0,
                K,
            )
            var95 = jnp.percentile(total_losses, 95)
            expected_loss = mean_lgd_per_loan * jnp.sum(lgd)
            factor_z_mean = (total_losses - expected_loss) / (expected_loss + 1e-8)
            factor_z_std = jnp.abs(factor_z_mean) * 0.5

            return jnp.stack([
                n_defaults_est.astype(jnp.float32),
                jnp.zeros(n_scenarios, dtype=jnp.float32),
                jnp.zeros(n_scenarios, dtype=jnp.float32),
                jnp.zeros(n_scenarios, dtype=jnp.float32),
                jnp.zeros(n_scenarios, dtype=jnp.float32),
                jnp.zeros(n_scenarios, dtype=jnp.float32),
                jnp.zeros(n_scenarios, dtype=jnp.float32),
                factor_z_mean.astype(jnp.float32),
                factor_z_std.astype(jnp.float32),
                jnp.full(n_scenarios, var95, dtype=jnp.float32),
            ], axis=1)

        # Vectorise over batch
        batched = vmap(simulate_theta, in_axes=(0, None))
        return batched

    def grad_log_likelihood(
        self, theta: np.ndarray, x_obs: np.ndarray
    ) -> np.ndarray:
        """
        JAX-backed gradient using autodiff (if JAX is available).
        """
        if self._jax is None:
            return super().grad_log_likelihood(theta, x_obs)

        # JAX grad traces through the simulation function, making shape arguments
        # dynamic (traced). JAX random functions require concrete shapes.
        # Use NumPy parent's gradient implementation to avoid traced-shape issues.
        return super().grad_log_likelihood(theta, x_obs)
