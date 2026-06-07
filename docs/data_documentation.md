# `data/` — Data layer

> Synthetic portfolio generation, stress-regime perturbations, and
> the (stubbed) real-data loader.

## Files

| File | Public API | Purpose |
|---|---|---|
| [`data/synthetic.py`](#syntheticpy) | `SyntheticPortfolioGenerator`, `PortfolioDataset` | Synthetic apartment loan portfolio with known θ |
| [`data/stress_regimes.py`](#stress_regimespy) | `StressRegimeGenerator`, `REGIME_SPECS` | Parametric stress shocks to θ |
| [`data/loader.py`](#loaderpy) | `RealDataLoader` | Real-data loader (stub) |

The package is exposed via `from data import ...` with the
following top-level names:

```python
from data.synthetic import SyntheticPortfolioGenerator, PortfolioDataset
from data.stress_regimes import StressRegimeGenerator, REGIME_SPECS
from data.loader import RealDataLoader
```

---

## `synthetic.py`

**Class** `SyntheticPortfolioGenerator(K: int = 10, seed: int | None = None)`.

Generates Finnish-style apartment loan portfolios with a known
ground-truth parameter vector `θ`. Uses a one-factor
**probit-link Gaussian** model (not the copula layer in
`copula/`). The generator is what the SBI / forward-simulator
pipeline uses to draw `(θ, x)` training pairs.

### `REGIONS = ("Helsinki", "Tampere", "Turku", "Oulu", "Other")`

Five fixed regions. The generator assigns loans to regions
deterministically based on factor loadings, so the same θ always
produces the same loan-region assignment.

### `PortfolioDataset` (frozen dataclass)

Container for a single simulation draw:

```python
@dataclass(frozen=True)
class PortfolioDataset:
    losses: np.ndarray             # float32[n_scenarios]
    observations: np.ndarray       # float32[n_scenarios, 10]
    theta: np.ndarray              # float32[D]
```

The `observations` matrix is a 10-column summary vector with
columns:

| # | Column | Definition |
|---|---|---|
| 0 | n_defaults | mean number of defaulted loans across scenarios |
| 1 | mean_lgd | (placeholder, always 0) |
| 2 | std_lgd | (placeholder, always 0) |
| 3 | helsinki_rate | mean fraction of Helsinki-region defaults |
| 4 | tampere_rate | mean fraction of Tampere-region defaults |
| 5 | turku_rate | mean fraction of Turku-region defaults |
| 6 | oulu_rate | mean fraction of Oulu-region defaults |
| 7 | factor_z_mean | mean of factor Z proxy across scenarios |
| 8 | factor_z_std | std of factor Z proxy across scenarios |
| 9 | var95 | 95th percentile of portfolio loss |

The forward simulator
([`simulator/forward.py`](simulator_documentation.md)) consumes
this vector verbatim.

### Methods

#### `sample(theta, n_scenarios) → PortfolioDataset`

The main entry point. Validates `theta.shape == (2K+4,)`,
derives per-loan LGD / principal / region from `theta` (via
`_derive_loan_attributes`), then runs `_simulate_losses` to
produce aggregated losses, then `losses_to_observation` to
produce the 10-column summary.

#### `losses_to_observation(losses) → np.ndarray[10]`

Aggregates a 1D losses array into the 10-column summary
described above. The regional default rates are computed as
`losses / region_exposure`, clipped to `[0, 1]`.

#### `_simulate_losses(theta, n_scenarios) → np.ndarray[n_scenarios]`

Core MC loop:

1. Sample systemic factor `Z` of shape `(n_scenarios,)`. If
   `tail_dep > 1e-6` and `nu < 100`, sample `Z` as a
   Student-t by mixture: `W / sqrt(V/nu)` with
   `W ~ N(0,1), V ~ χ²(nu)`. Otherwise `Z ~ N(0,1)`.
2. Compute conditional default probabilities via the probit
   link: `p_i | Z = Φ((Φ⁻¹(p_i) - sqrt(ρ)·b_i·Z) / sqrt(1 - ρ·b_i²))`.
3. Draw Bernoulli defaults `U < p_i | Z`.
4. Per-loan losses: `lgd_i · principal_i · default_i`.
5. Aggregate: `losses = sum_i loan_loss_i`.

The vectorised implementation uses NumPy broadcasting; for K=10
and n_scenarios=1_000_000, runtime is ~50 ms per call.

#### `_derive_loan_attributes(theta) → None`

Sets `self._lgd`, `self._principal`, `self._region_idx` based
on `theta`:

- `lgd`: uniform in `[0.20, 0.60]`, plus 0.05 if the factor
  loading is negative (stress-sensitive loans), clipped to
  `[0.10, 0.80]`.
- `principal`: log-uniform in `[€50k, €500k]`.
- `region_idx`: assigned by ranking of factor loadings using a
  fixed array `[0, 0, 0, 1, 1, 2, 2, 3, 3, 4]` (only the first
  K entries are used).

These attributes are **fixed for a given θ**, so calling
`sample` multiple times with the same θ produces
identically-distributed loans but different scenarios.

---

## `stress_regimes.py`

**Class** `StressRegimeGenerator(seed: int | None = None)`.

Applies parametric shocks to a baseline `θ` to produce stressed
`θ_perturbed` for each of five named regimes.

### `REGIME_SPECS`

A dict from regime name to a `(theta, shock, K) → theta_perturbed`
function:

| Regime | Effect |
|---|---|
| `baseline` | identity (no shock) |
| `housing_crash` | scale p_zeros by 1.3×–2.0×, amplify negative loadings, +0.5·shock on tail_dep, +0.20·shock on ρ |
| `rate_shock` | scale p_zeros by 1.5×–4.0×, +0.15·shock on ρ, +0.3·shock on tail_dep |
| `unemployment` | double p_zeros for the bottom half of factor loadings, scale loadings down by 15%, +0.25·shock on ρ |
| `liquidity` | +0.4·shock on tail_dep, -2·shock on ν (down to floor 3.0), +0.15·shock on ρ |

The shock magnitude is in `[0, 1]`; `0 = no shock`,
`1 = full stress`. Each function:

1. Returns `theta.copy()` if `shock <= 0`.
2. Otherwise modifies the relevant slots of the copy in-place
   (after a `theta.copy()`) and clips to physical bounds.

### `StressRegimeGenerator.sample(regime_name, theta_baseline, shock_magnitude=1.0) → np.ndarray`

Validates the regime name (raises `ValueError` for unknown
regimes), infers `K` from `len(theta_baseline)`, then dispatches
to the corresponding function in `REGIME_SPECS`. Returns the
perturbed `θ_perturbed` as a fresh `float32` array.

### Mathematical notes

- **Housing crash** amplifies p_zeros and loadings, modelling
  the joint effect of falling house prices and increasing
  default sensitivity. Tail dependence and correlation both
  increase, capturing the empirical observation that defaults
  become more co-moving in downturns.
- **Rate shock** scales p_zeros more aggressively than housing
  crash, capturing the higher direct impact of interest-rate
  increases on PDs. Correlation rises via credit-market stress.
- **Unemployment** targets the *most exposed* loans (bottom half
  of factor loadings) and shrinks the loadings themselves,
  reflecting the bank's view that unemployment is a
  region-specific shock.
- **Liquidity** modifies tail_dep and ν (not p_zeros), modelling
  the *recovery* side of loss — when markets are illiquid, LGDs
  rise even if PDs are unchanged.

### Usage example

```python
import numpy as np
from data.synthetic import SyntheticPortfolioGenerator
from data.stress_regimes import StressRegimeGenerator

K = 10
gen = SyntheticPortfolioGenerator(K=K, seed=42)
stress = StressRegimeGenerator(seed=42)

theta_baseline = np.array(
    [0.3] * K + [0.02] * K + [0.0, 0.5, 30.0, 0.0], dtype=np.float32
)
theta_stressed = stress.sample("housing_crash", theta_baseline, shock_magnitude=1.0)

dataset = gen.sample(theta_stressed, n_scenarios=10_000)
print(dataset.observations.shape)  # (10000, 10)
```

---

## `loader.py`

**Class** `RealDataLoader(source: str = "statfin")`.

**Stub** for the real Finnish apartment loan data loader. Calling
`fetch()` raises `NotImplementedError` with a clear message and
pointers to:

1. **StatFin** (`stat.fi`) — apartment price index, mortgage
   rate, household debt.
2. **Eurostat** — regional unemployment, construction permits.
3. **Bank of Finland** (`suomenpankki.fi`) — household credit,
   NPL ratios.

The real-data path is a Phase 1 extension (architecture
decision D6). Until real access is confirmed, the
**synthetic generator is the primary data source** (per the
architecture doc).

The expected return signature (documented in the module
docstring) is:

```python
features:   np.ndarray[float32[N, 4]]   # [PD, LGD, EAD, maturity]
defaults:   np.ndarray[bool[N]]
lgd:        np.ndarray[float32[K]]      # unique LGD values
```
