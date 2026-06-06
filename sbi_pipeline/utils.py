"""SBI Training Utilities for QSVT4CRA.

Provides configuration dataclasses, prior helpers, and W&B logging hooks
for the SBI training pipelines.

Memory note (from prior research):
- sbi 0.22.0: `from sbi.utils import posterior_nn`
- `flow._prior` -> `flow.prior` (deprecated API fix)
- CpuPrior wrapper for `.log_prob()` and `.sample()` compatibility
- `n_workers=1` for JAX thread safety
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.distributions as dist
from sbi.utils import BoxUniform

# Optional imports with graceful fallback
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class SBITrainingConfig:
    """Configuration for SBI training pipelines.

    Parameters
    ----------
    prior
        Prior distribution over parameters (e.g., BoxUniform).
    n_rounds
        Number of training rounds (proposal adaptation rounds).
    n_simulations_per_round
        Number of simulations per round.
    batch_size
        Batch size for training.
    learning_rate
        Learning rate for optimization.
    hidden_features
        Number of hidden features in the neural density estimator.
    num_transforms
        Number of inverse autoregressive flow transforms.
    device
        Device for training ('cuda' or 'cpu').
    wandb_project
        Optional W&B project name for logging.
    seed
        Random seed for reproducibility.
    """

    prior: dist.Distribution
    n_rounds: int = 10
    n_simulations_per_round: int = 1000
    batch_size: int = 100
    learning_rate: float = 5e-4
    hidden_features: int = 50
    num_transforms: int = 4
    device: str = "cuda"
    wandb_project: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {self.n_rounds}")
        if self.n_simulations_per_round < 1:
            raise ValueError(
                f"n_simulations_per_round must be >= 1, "
                f"got {self.n_simulations_per_round}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.hidden_features < 1:
            raise ValueError(
                f"hidden_features must be >= 1, got {self.hidden_features}"
            )
        if self.num_transforms < 1:
            raise ValueError(
                f"num_transforms must be >= 1, got {self.num_transforms}"
            )
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")


def get_prior_from_bounds(
    low: np.ndarray,
    high: np.ndarray,
) -> BoxUniform:
    """Create a uniform prior from lower and upper bounds.

    Parameters
    ----------
    low
        Lower bounds for each parameter, shape (D,).
    high
        Upper bounds for each parameter, shape (D,).

    Returns
    -------
    sbi.utils.BoxUniform prior distribution.

    Raises
    ------
    ValueError
        If low >= high for any parameter dimension.

    Examples
    --------
    >>> import numpy as np
    >>> low = np.array([0.0, 0.0, 0.0])
    >>> high = np.array([1.0, 2.0, 0.5])
    >>> prior = get_prior_from_bounds(low, high)
    >>> samples = prior.sample((100,))
    """
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    if low.shape != high.shape:
        raise ValueError(
            f"low and high must have the same shape, "
            f"got {low.shape} and {high.shape}"
        )

    if np.any(low >= high):
        raise ValueError(
            f"All low bounds must be strictly less than high bounds. "
            f"Got low={low}, high={high}"
        )

    return BoxUniform(low=torch.from_numpy(low), high=torch.from_numpy(high))


class WandbLoggingHook:
    """W&B logging hook for SBI training.

    Provides logging hooks that are called during training to log metrics
    to W&B. If WANDB_API_KEY is not set, this becomes a no-op.

    Parameters
    ----------
    project
        W&B project name.
    name
        Run name.
    config
        Optional config dict to log.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self._project = project
        self._name = name
        self._config = config or {}
        self._run = None

    def __enter__(self):
        """Start W&B run on context entry."""
        if WANDB_AVAILABLE:
            self._run = wandb.init(
                project=self._project,
                name=self._name,
                config=self._config,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG001
        """Finish W&B run on context exit."""
        if self._run is not None:
            self._run.finish()

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Parameters
        ----------
        metrics
            Dict of metrics to log.
        step
            Optional step number.
        """
        if self._run is not None:
            wandb.log(metrics, step=step)

    def log_summary(self, summary: dict) -> None:
        """Log summary metrics to W&B.

        Parameters
        ----------
        summary
            Dict of summary metrics.
        """
        if self._run is not None:
            for key, value in summary.items():
                wandb.run.summary[key] = value


class NoOpLoggingHook:
    """No-op logging hook when W&B is not available."""

    def __init__(self, project: str = None, name: str = None, config: dict = None):  # noqa: ARG002
        """Initialize no-op hook (ignores all arguments)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG001
        pass

    def log(self, metrics: dict, step: Optional[int] = None) -> None:  # noqa: ARG001
        """No-op log."""
        pass

    def log_summary(self, summary: dict) -> None:  # noqa: ARG001
        """No-op log summary."""
        pass


def get_logging_hook(
    project: Optional[str],
    name: Optional[str] = None,
    config: Optional[dict] = None,
) -> WandbLoggingHook | NoOpLoggingHook:
    """Get appropriate logging hook based on W&B availability.

    Parameters
    ----------
    project
        W&B project name.
    name
        Optional run name.
    config
        Optional config dict.

    Returns
    -------
    WandbLoggingHook if W&B available, else NoOpLoggingHook.
    """
    if WANDB_AVAILABLE and project is not None:
        return WandbLoggingHook(project=project, name=name, config=config)
    return NoOpLoggingHook()
