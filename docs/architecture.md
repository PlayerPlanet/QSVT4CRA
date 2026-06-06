# QSVT4CRA — Full System Architecture

**Project**: Posterior-Propagated Factor-Copula QSVT for Apartment Loan Portfolio Risk  
**Date**: 2026-06-06  
**Compute**: LUMI-G MI250X (50–150 GPU-hours, <500 CPU-hours target)  
**Status**: Design v1.0 — For implementation planning

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        QSVT4CRA Research Run — Data Flow                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Apartment Loan Data
         │
         ▼
  ┌─────────────────────────────┐
  │   Stress Regime Generator   │  ← θ_ground_truth (parametric shocks)
  └─────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │     Forward Simulator       │  → synthetic observations x
  │        (JAX/NumPy)          │
  └─────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │   SBI Posterior Estimator    │  → p(θ | data_obs) via NPE/NLE/flow-matching
  │       (PyTorch + sbi)       │
  └─────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │     Posterior Sampler       │  → {θ⁽¹⁾, …, θ⁽ᴺ⁾} (N=1000–10000)
  │    (sbi.build_posterior)    │
  └─────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                        PER-POSIERIOR-SAMPLE LOOP                                │
  │  ┌──────────────────────────────┐    ┌───────────────────────────────────────┐  │
  │  │   Factor Copula Simulator    │    │   PosteriorFactorCopulaLoader       │  │
  │  │   (Gaussian / Student-t /    │    │   (replaces MultivariateGCI)         │  │
  │  │    D-vine)                  │    │   → quantum state-prep U(θ⁽ⁱ⁾)        │  │
  │  └──────────────────────────────┘    └───────────────────────────────────────┘  │
  │                    │                                   │                        │
  │                    ▼                                   ▼                        │
  │           Portfolio Losses                     Qiskit QuantumCircuit          │
  │              {L_j}                                    U(θ)                      │
  │                    │                                   │                        │
  │                    └──────────┬────────────────────────┘                        │
  │                               ▼                                                 │
  │                   ┌─────────────────────────┐                                    │
  │                   │  QSVT Polynomial        │  ← pyqsp 0.2.0 phases              │
  │                   │  Approximator           │    + threshold mapping            │
  │                   └─────────────────────────┘                                    │
  │                               │                                                 │
  │                               ▼                                                 │
  │                   ┌─────────────────────────┐                                    │
  │                   │  QSVTRiskCircuit         │  ← VaR/CVaR measurement           │
  │                   │  (full circuit)          │    P(loss ≤ target)              │
  │                   └─────────────────────────┘                                    │
  │                               │                                                 │
  │                               ▼                                                 │
  │                    Per-sample VaR/CVaR ← measurement result                     │
  └──────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │        Aggregator           │  → posterior-predictive VaR/CVaR
  │   (numpy / xarray)          │    + uncertainty bands (5th/95th)
  └─────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │      Validation Gate        │  → classical MC ground truth comparison
  │   (metrics/quantum_error)   │    + SBC posterior coverage check
  └─────────────────────────────┘
```

### Mermaid Equivalent

```mermaid
flowchart LR
    A[Apartment Loan Data] --> B[Stress Regime Generator]
    B --> C[Forward Simulator]
    C --> D[SBI Posterior Estimator]
    D --> E[Posterior Sampler]
    E --> F{For each θ⁽ⁱ⁾}
    F --> G[Factor Copula Simulator]
    F --> H[PosteriorFactorCopulaLoader]
    G --> I[Portfolio Losses]
    H --> J[Qiskit QuantumCircuit Uθ]
    I --> K[QSVT Polynomial Approximator]
    J --> K
    K --> L[QSVTRiskCircuit]
    L --> M[Per-sample VaR/CVaR]
    M --> N[Aggregator]
    N --> O[Posterior-predictive VaR/CVaR]
    O --> P[Validation Gate]
```

---

## 2. Module Decomposition

| Module | Purpose | Input | Output | Backend |
|--------|---------|-------|--------|---------|
| `data/loader.py` | Finnish apartment loan data ingestion | CSV/parquet path | `(features: float32[N,4], defaults: bool[N], lgd: float32[K])` | CPU pandas |
| `data/synthetic.py` | Synthetic portfolio generator with known θ | regime spec dict | `(portfolio, θ_ground_truth)` | NumPy |
| `data/stress_regimes.py` | Parametric stress shock generator | regime_type, shock_magnitude | `θ_perturbed` | NumPy |
| `simulator/forward.py` | Forward model: θ → observations | `θ: float32[D]` | `x: float32[T]` (T time steps) | JAX on GPU |
| `sbi/posterior.py` | NPE/NLE/flow-matching SBI training | `{(θᵢ, xᵢ)}` training pairs | `Posterior` object + sampler | PyTorch + sbi 0.22.0 |
| `sbi/utils.py` | SBI training helpers (prior, proposal, logging) | config dict | W&B artifact paths | PyTorch + W&B |
| `copula/gaussian.py` | Gaussian factor copula simulator | `θ: float32[D]`, `n_samples: int` | `(U_marginals: float32[n_samples, K], losses: float32[n_samples])` | NumPy |
| `copula/student_t.py` | Student-t factor copula simulator | `θ + dof: float` | `(U_marginals, losses)` | NumPy |
| `copula/vine.py` | D-vine copula simulator | `θ`, `structure: list` | `(U_marginals, losses)` | pyvine or hand-rolled |
| `copula/low_rank.py` | Low-rank factor copula (SBI baseline) | `θ`, `rank: int` | `(U_marginals, losses)` | NumPy |
| `loader/posterior_factor_copula.py` | `PosteriorFactorCopulaLoader` — **replaces `MultivariateGCI`** | `θ⁽ⁱ⁾: float32[D]` | `QuantumCircuit` (amplitude loading) | Qiskit |
| `loader/amplitude_loader.py` | Optimized amplitude loading for large K | `losses: float32[2^K]`, `target_loss` | `QuantumCircuit` | Qiskit |
| `qsvt/approximator.py` | QSVT polynomial phase computation | `target_loss: float`, `degree: int`, `poly_type: str` | `phases: list[float]` | pyqsp 0.2.0 |
| `qsvt/circuit.py` | `QSVTRiskCircuit` — full risk circuit | `θ⁽ⁱ⁾`, `target_loss`, `degree` | `QuantumCircuit` | Qiskit |
| `qsvt/threshold.py` | Threshold function construction | `threshold: float`, `degree` | `poly: list[float]` | NumPy |
| `metrics/var_cvar.py` | Classical VaR/CVaR from samples | `losses: float32[N]` | `dict(VaR_95, VaR_99, CVaR_95, CVaR_99)` | NumPy |
| `metrics/quantum_error.py` | Quantum vs classical error analysis | `circuit`, `classical_value` | `dict(error_L_inf, error_L_1, error_L_2)` | Qiskit Aer |
| `metrics/coverage.py` | SBC coverage check for posterior | `θ_samples`, `true_θ` | `coverage_percentage` | NumPy |
| `experiments/sbi_train.py` | CLI: SBI training entry point | `--config path` | `checkpoints/` | All |
| `experiments/mc_ground_truth.py` | CLI: MC ground truth generation | `--n_scenarios`, `--posterior` | `results.npz` | All |
| `experiments/qsvt_sweep.py` | CLI: QSVT degree sweep | `--degrees`, `--posterior` | `results.npz` | All |
| `experiments/ood_robustness.py` | CLI: OOD robustness test | `--regimes` | `results.npz` | All |
| `experiments/resource_scaling.py` | CLI: resource scaling study | `--n_loans` | `results.npz` | All |
| `experiments/make_figures.py` | CLI: figure generation | `--results_dir` | `figures/*.png` | matplotlib |
| `configs/default.yaml` | Default config (Hydra) | — | — | YAML |
| `configs/sbi_train.yaml` | SBI training config | — | — | YAML |
| `configs/stress_regimes.yaml` | Stress regime definitions | — | — | YAML |
| `configs/quantum.yaml` | Quantum circuit config (degrees, backend) | — | — | YAML |

### Swapability Notes

- **`MultivariateGCI_*` ↔ `PosteriorFactorCopulaLoader`**: Drop-in replacement. Both export a `QuantumCircuit` with `.num_qubits` and `.to_gate()`. The loader API must match the existing call signature in `circuitsCRA.py` (`get_expected_probability_circuit`).
- **SBI backend**: `sbi` 0.22.0 is the primary inference engine. Can swap to `nflows` or `pyro` if needed, but the `Posterior` interface must be preserved.
- **Factor copula variants**: All inherit from `BaseFactorCopula` interface with `.sample(θ, n)` and `.losses(U_marginals)` methods.

---

## 3. Key Design Decisions

### D1: SBI Estimator Choice — NPE as Primary, Flow-Matching as Fallback

**Decision**: Use **Neural Posterior Estimation (NPE)** as the primary SBI estimator.

**Rationale**:
- NPE is the most mature SBI method in `sbi` 0.22.0, with proven convergence for continuous parameter spaces.
- It provides a full posterior `p(θ|x)` (not just point estimate), enabling posterior-predictive uncertainty propagation.
- NPE scales well to 10–50 dimensional θ spaces (factor loadings + default thresholds + tail dependence).
- `sbi` 0.22.0 has a stable `NeuralPosterior` API with `.sample()` and `.log_prob()`.

**Alternatives considered**:
- **NLE (Neural Likelihood Estimation)**: Requires a separate likelihood model; adds complexity. Discarded.
- **NSM-Bayes**: Mentioned in `Memory/Projects/quantumhack.md` but lacks public documentation. Harder to debug. Fallback only.
- **Flow matching**: Modern, potentially better for heavy-tailed posteriors. Not yet in `sbi` 0.22.0 mainline. Monitor as upgrade path.

**Fallback**: If NPE ESS < 200 after 50k training steps, switch to NLE or flow matching.

---

### D2: Factor-Copula Structure — Gaussian Primary + Student-t Fallback

**Decision**: Implement **Gaussian factor copula** as primary, **Student-t factor copula** as secondary for heavy tails.

**Rationale**:
- Gaussian factor copula is the natural extension of `MultivariateGCI_Linear`, making the loader replacement straightforward.
- It is computationally tractable: `O(K·n_samples)` for K assets and n samples, no iterative optimization.
- Student-t adds one parameter (degrees of freedom) and handles tail dependence, critical for stress regimes.
- D-vine is complex to implement and fit; reserved for Phase 2 extension if time permits.

**Alternatives considered**:
- **D-vine copula**: Most flexible but requires sequential fitting and structure selection. High combinatorial cost. Fallback for Phase 2 extension.
- **Low-rank factor copula**: Mentioned in `Memory/Projects/trust-crisis-sbi.md`. Good for scalability but adds implementation complexity. Use if Gaussian shows insufficient tail modeling.

**Fallback**: If Gaussian factor copula fails SBC coverage check (coverage < 90% for any parameter), switch to Student-t.

---

### D3: Mapping θ⁽ⁱ⁾ → Quantum Circuit — New `PosteriorFactorCopulaLoader`

**Decision**: Build a **new `PosteriorFactorCopulaLoader`** class that replaces `MultivariateGCI_Poly`/`MultivariateGCI_Linear` entirely.

**Rationale**:
- `MultivariateGCI_*` encodes factor structure via `NormalDistribution` (2^n qubits for n latent factors) — not scalable to 1000 loans.
- `PosteriorFactorCopulaLoader` will use **amplitude encoding** of the loss distribution: the circuit encodes `|j⟩ → |L_j⟩` where `L_j` is the portfolio loss for scenario j.
- This directly feeds into `get_expected_probability_circuit` which expects an amplitude-loading-style uncertainty model.
- The existing `AmplitudeLoading` class can be reused/extended for the new loader.

**Implementation approach**:
1. For each posterior sample θ⁽ⁱ⁾, run factor copula MC to get loss distribution `{L_j}` for j=0..2^K-1.
2. Construct `AmplitudeLoadingVar` (or new class) with scaled angles proportional to `arcsin(L_j / max_loss)`.
3. This becomes the `uncertainty_model` input to `get_expected_probability_circuit`.

**Note**: `Memory/Projects/quantumhack.md` mentions `BlockEncoder` with diagonal live + `_encode_general` placeholder — this may be a useful reference for efficient encoding.

---

### D4: QSVT Polynomial Construction — pyqsp 0.2.0 + Chebyshev Fallback

**Decision**: Use **pyqsp 0.2.0** for phase sequence computation, with **Chebyshev polynomial** basis.

**Rationale**:
- `pyqsp==0.2.0` is the pinned version in `requirements.txt` — must use it.
- The comment in `circuitsCRA.py` ("only old versions with low degree polynomials approximations work") indicates degree limits. Likely max ~64–128.
- `QuantumSignalProcessingPhases(poly, signal_operator="Wx", method="sym_qsp", measurement="x", chebyshev_basis=True)` is already used in `QSVT.py`.
- `adjust_conventions=True` handles the QSP→QSVT convention conversion (already implemented in `QSVT.adjust_qsvt_convetions`).

**Degree sweep plan**: `[16, 32, 64]` (primary) → `[128, 256]` (if pyqsp permits) → `[512, 1024]` (if time permits).

**Alternatives considered**:
- **Qiskit `qiskit_algorithms` QSVT**: Not in requirements.txt. Would require new dependency. Not critical path.
- **Hand-rolled Chebyshev**: Possible fallback if pyqsp fails at high degree. Implement in `qsvt/threshold.py`.
- **Piecewise approximation**: If max degree insufficient, split threshold function into segments. Phase 4 extension.

**Critical constraint**: pyqsp 0.2.0 degree limits must be verified experimentally before Phase 4. Do not assume >64 works.

---

### D5: Ground-Truth MC Scale — 1e6 Scenarios Primary, 1e7 for Tail Metrics

**Decision**: Use **1e6 scenarios** per posterior sample for standard metrics; **1e7** for tail VaR/CVaR (99.9th percentile).

**Rationale**:
- 1e6 scenarios → ~0.1% Monte Carlo error for VaR₉₅. Acceptable for initial experiments.
- 1e7 scenarios → ~0.03% error for VaR₉₉.₉. Needed for 99.9th percentile validation.
- CPU-hours budget: 1e6 scenarios × 1000 loans × 5 regimes ≈ 200–400 CPU-hours (within <500 budget).
- GPU not needed for MC; reserve GPU hours for Qiskit Aer simulation.

**Validation protocol**:
- For Gaussian case, validate MC against analytical CDF (known closed form).
- For Student-t, use `scipy.stats.multivariate_t` CDF as reference.
- Report confidence intervals via bootstrapping (1000 resamples).

**Alternatives considered**:
- **1e8 scenarios**: Would exceed CPU budget (2000+ CPU-hours). Not feasible.
- **Stratified sampling**: Could reduce variance but adds implementation complexity. Phase 3 extension if time.

---

### D6: Finnish Apartment Loan Data — Synthetic Generator Primary

**Decision**: Use **synthetic data generator** as primary; attempt real data from StatFin/Eurostat as Phase 1 extension.

**Rationale**:
- Synthetic generator provides full control over ground-truth θ, enabling end-to-end validation.
- Real Finnish housing data (StatFin/Eurostat) requires API integration, licensing review, and data cleaning — could delay Phase 1.
- Synthetic generator can match marginal distributions (PD, LGD) from published Finnish banking statistics.

**Implementation**:
- `data/synthetic.py` generates portfolios with known factor loadings, default thresholds, and tail dependence.
- Regime types: baseline, housing crash, rate shock, unemployment spike, liquidity crisis.
- Ground-truth θ logged and used for SBC validation.

**Real data path** (Phase 1 extension):
- Attempt StatFin API (`stat.fi`) for Finnish apartment price index + interest rate data.
- Eurostat for household debt-to-income ratios.
- If accessible, use real data for forward simulator training.

**Critical-path flag**: If real data access is blocked, synthetic fallback must be ready by Week 2.

---

### D7: Stress Regime Generation — Parametric Shock + Copula Reparameterization

**Decision**: Use **parametric shock** applied to factor-copula parameters, with regime-dependent reparameterization.

**Rationale**:
- Each stress regime (housing crash, rate shock, etc.) maps to specific parameter perturbations: e.g., housing crash → increase factor loadings on housing-sensitive assets.
- This is consistent with regulatory stress testing frameworks (EBA, FIN-FSA).
- The same factor-copula structure (Gaussian/Student-t) applies across regimes; only θ changes.
- Enables SBI training on broad regime data and testing on stressed regimes.

**Regime definitions** (from `linear_backlog.md`):
| Regime | Factor Shock | Copula Parameter Change |
|--------|-------------|------------------------|
| Baseline | None | θ₀ |
| Housing crash | Housing factor loadings × 1.5 | ρ increases 0.2 |
| Rate shock | Rate factor loadings × 2.0 | ρ increases 0.15 |
| Unemployment | Unemployment factor loadings × 1.8 | ρ increases 0.25 |
| Liquidity crisis | All correlations × 1.3 | dof decreases 2 |

**Alternatives considered**:
- **Non-parametric regime discovery**: Use clustering on historical data. Adds complexity. Not needed for initial experiments.
- **Mixture model**: Blend regimes with weights. Useful for posterior pooling but not critical path.

---

## 4. Data Flow Contracts

### 4.1 Module Boundary Contracts

```
data/loader.py
───────────────
Input:  path: str (CSV/parquet)
Output: features: float32[N, 4]     # [PD, LGD, EAD, maturity]
        defaults: bool[N]
        lgd: float32[K]
Shapes: features.shape = (N_loans, 4)
Dtype: float32, bool
Notes: K = number of unique LGD values (typically 1 for homogeneous portfolio)

data/synthetic.py
─────────────────
Input:  regime_spec: dict with keys {n_loans, n_scenarios, regime_type, seed}
Output: portfolio: dict(features, defaults, lgd)
        θ_ground_truth: float32[D]  # [factor_loadings, thresholds, rho, dof]
Shapes: portfolio['features'].shape = (n_loans, 4)
        θ_ground_truth.shape = (D,) where D ≤ 20
Dtype: float32

simulator/forward.py
────────────────────
Input:  θ: float32[D]
        T: int (time steps)
Output: x: float32[T]  # observation time series
Shapes: x.shape = (T,) or (T, n_features)
Dtype: float32

sbi/posterior.py
─────────────────
Input:  training_pairs: list of (θᵢ: float32[D], xᵢ: float32[T]) tuples
        n_training: int (default 10000)
Output: posterior: sbi.utils.posterior_nn neural network
        sampler: callable (n_samples) → θ_samples: float32[N, D]
Shapes: θ_samples.shape = (n_samples, D)
Dtype: float32

copula/gaussian.py
──────────────────
Input:  θ: float32[D]  # [rhos, factor_loadings, default_thresholds]
        n_samples: int
Output: U_marginals: float32[n_samples, K]  # uniform marginals [0,1]
        losses: float32[n_samples]           # portfolio losses
Shapes: U_marginals.shape = (n_samples, K)
        losses.shape = (n_samples,)
Dtype: float32
Notes: K = number of loans (assets)
       losses computed via inverse CDF + aggregation

loader/posterior_factor_copula.py
─────────────────────────────────
Input:  θ: float32[D]
        K: int (number of loans)
        max_loss: float32
Output: circuit: QuantumCircuit
        num_qubits: int
        num_ancillas: int
Shapes: circuit.num_qubits = K + 1 (target qubit) + ancillas
Dtype: N/A
Notes: Implements amplitude loading of loss distribution
       Compatible with get_expected_probability_circuit interface

qsvt/approximator.py
─────────────────────
Input:  target_loss: float32 (in [0, max_loss])
        degree: int
        poly_type: str ("chebyshev" | "legendre")
Output: phases: list[float] (length = degree)
        poly_coeffs: list[float]
Shapes: len(phases) = degree
Dtype: float32
Notes: Uses pyqsp.angle_sequence.QuantumSignalProcessingPhases
       poly_coeffs from threshold function approximation

qsvt/circuit.py
───────────────
Input:  θ⁽ⁱ⁾: float32[D]
        target_loss: float32
        degree: int
        lgd: float32[K]
Output: circuit: QuantumCircuit (full QSVT risk circuit)
        objective_circuit: QuantumCircuit (amplitude loading sub-circuit)
Shapes: circuit.num_qubits = K + 1 + ancillas
Dtype: N/A
Notes: Composes PosteriorFactorCopulaLoader + QSVT phases

metrics/var_cvar.py
───────────────────
Input:  losses: float32[N]
        alphas: list[float] (e.g., [0.95, 0.99, 0.999])
Output: metrics: dict
        {
          'VaR_95': float, 'VaR_99': float, 'VaR_999': float,
          'CVaR_95': float, 'CVaR_99': float, 'CVaR_999': float,
          'tail_prob': float (P(loss > threshold))
        }
Shapes: losses.shape = (N,)
Dtype: float32
Notes: VaR_α = np.quantile(losses, α)
       CVaR_α = mean(losses[losses >= VaR_α])

metrics/quantum_error.py
───────────────────────
Input:  circuit: QuantumCircuit
        classical_value: float (true CDF or probability)
        target_loss: float
        n_shots: int (default 10000)
Output: error_metrics: dict
        {
          'error_L_inf': float,
          'error_L_1': float,
          'error_L_2': float,
          'quantum_value': float,
          'n_shots': int
        }
Shapes: scalar outputs
Dtype: float32
Notes: Uses Qiskit Aer simulator for measurement sampling
```

### 4.2 End-to-End Tensor Shapes

```
Full pipeline per posterior sample:
────────────────────────────────────
θ⁽ⁱ⁾: float32[D] (D ≤ 20)
  ↓
copula.sample(θ⁽ⁱ⁾, n=1_000_000) → losses: float32[1_000_000]
  ↓
PosteriorFactorCopulaLoader(θ⁽ⁱ⁾) → circuit: QuantumCircuit (K+1 qubits)
  ↓
QSVT(circuit, phases) → full_circuit: QuantumCircuit
  ↓
Aer.simulate(full_circuit, shots=10_000) → measurement_counts: dict
  ↓
post_processing(measurement_counts) → VaR/CVaR: float

Aggregator across N posterior samples:
──────────────────────────────────────
{VaR⁽¹⁾, ..., VaR⁽ᴺ⁾}: float32[N]
  ↓
posterior_mean = mean(VaR⁽ⁱ⁾)
posterior_std = std(VaR⁽ⁱ⁾)
uncertainty_bands = [percentile(VaR, 5), percentile(VaR, 95)]
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

| Module | Test | Validation Criterion |
|--------|------|---------------------|
| `data/synthetic.py` | `test_portfolio_shape`, `test_θ_recoverability` | Output shapes match spec; known θ recoverable via SBC |
| `simulator/forward.py` | `test_output_dtype`, `test_JAX_on_GPU` | float32 output; runs on MI250X |
| `sbi/posterior.py` | `test_ess_convergence`, `test_log_prob_improves` | ESS > 200; test log-likelihood > prior by 2 nats |
| `copula/gaussian.py` | `test_cdf_matches_scipy`, `test_tail_symmetry` | CDF error < 1e-3 vs `scipy.stats.multivariate_normal` |
| `copula/student_t.py` | `test_dof_effect`, `test_tail_dependence` | CDF error < 1e-2 vs `scipy.stats.multivariate_t` |
| `loader/posterior_factor_copula.py` | `test_circuit_num_qubits`, `test_amplitude_loading_correctness` | num_qubits = K+1+ancillas; amplitude matches losses |
| `qsvt/approximator.py` | `test_phase_sequence_length`, `test_poly_degree` | len(phases) = degree; poly coefficients valid |
| `metrics/var_cvar.py` | `test_analytical_gaussian`, `test_quantile_monotonicity` | VaR₉₅ < VaR₉₉ < VaR₉₉.₉; CVaR ≥ VaR |

### 5.2 Property Tests

- **Copula symmetry**: Generated U_marginals must be uniform on [0,1] (Kolmogorov-Smirnov test, p > 0.01).
- **Posterior coverage via SBC**: 90% of true θ must lie within 90% credible intervals.
- **VaR monotonicity**: VaR(α) must be non-decreasing in α.
- **CVaR consistency**: CVaR(α) ≥ VaR(α) for all α.

### 5.3 Integration Tests

- **End-to-end on synthetic data**: Run full pipeline with known θ. Validate VaR error < 5% vs MC ground truth.
- **SBI → Factor copula → QSVT**: Test that posterior samples produce sensible loss distributions (no NaN, no extreme outliers).
- **Quantum vs classical**: For K ≤ 10, compare QSVT circuit output to classical CDF. Error must be < 1e-2.

### 5.4 Golden Tests

- **Gaussian copula VaR**: Analytical VaR₉₅ exists in closed form. Validate MC matches to 1e-3.
- **Fixed θ baseline**: When θ is fixed (no posterior), VaR/CVaR must match `scipy.stats` reference.
- **pyqsp phase sequence**: For degree=16, phases must match pre-computed reference (commit to repository).

---

## 6. Compute Estimate

### 6.1 Per-Phase Budget

| Phase | GPU-hours | CPU-hours | Wall-clock | Notes |
|-------|-----------|-----------|------------|-------|
| **Phase 1**: SBI Training | 20–40 | 10–20 | 8–16 hrs | PyTorch on MI250X; W&B logging |
| **Phase 2**: Factor Copula | 5–10 | 20–40 | 4–8 hrs | CPU NumPy; vectorized MC |
| **Phase 3**: MC Ground Truth | 0 | 200–400 | 1–2 hrs | CPU NumPy; 1e6–1e7 scenarios |
| **Phase 4**: QSVT Loader + Sweep | 30–80 | 50–100 | 4–12 hrs | Qiskit Aer on GPU; pyqsp fitting |
| **Phase 5**: OOD Robustness | 10–20 | 20–40 | 4–8 hrs | Re-training SBI on subset |
| **Phase 6**: Resource Scaling | 5–10 | 10–20 | 2–4 hrs | CPU; estimation only |
| **Phase 7**: Figures | 5–10 | 10–20 | 4–8 hrs | matplotlib rendering |
| **Total** | **75–170** | **320–640** | **27–54 hrs** | ⚠️ Exceeds 500 CPU-hour target |

### 6.2 CPU-Hour Reduction Strategy

If total CPU-hours exceed 500:
- **Phase 3**: Reduce from 1e7 to 1e6 scenarios for tail VaR₉₉.₉. Accept larger CI.
- **Phase 2**: Limit D-vine fitting to top 5 asset pairs (reduce combinatorial search).
- **Phase 4**: Reduce QSVT degree sweep to `[16, 64, 256, 1024]` (skip intermediate degrees).

### 6.3 GPU-Hour Allocation

Priority order:
1. **Phase 4 (QSVT)**: 30–80 GPU-hours — critical path, requires Qiskit Aer simulation.
2. **Phase 1 (SBI)**: 20–40 GPU-hours — foundational, enable all subsequent phases.
3. **Phase 5 (OOD)**: 10–20 GPU-hours — only if GPU budget permits.
4. **Phase 6–7**: 5–10 GPU-hours each — low priority, can run on CPU.

### 6.4 Memory Estimate

| Data Structure | Size | Notes |
|---------------|------|-------|
| 1 posterior sample × 1e6 losses | 4 MB | float32 |
| N=1000 posterior samples × 1e6 losses (on-the-fly) | 0 GB | Aggregate on the fly; do not store |
| SBI training set (10k × θ, x pairs) | 80 MB | PyTorch buffer |
| QSVT circuit (K=1000) | 100–500 MB | Depends on transpilation |
| **Total peak memory** | < 1 GB | Per job |

**Critical**: Storing 1e8 scenarios × 1000 loans = 800 GB is NOT feasible. All aggregation must be on-the-fly (streaming quantile computation).

---

## 7. Risk Register

| Risk ID | Description | Likelihood | Impact | Mitigation | Owner |
|---------|-------------|------------|--------|------------|-------|
| **R1** | pyqsp 0.2.0 degree limits (< 64–128) restrict high-accuracy QSVT approximations | High | Medium | Pre-compute max degree per accuracy target; use piecewise approximation if needed; test early in Phase 4 | ML-workflow |
| **R2** | Qiskit `NormalDistribution` qubit scaling (2^n qubits for n assets) makes `MultivariateGCI` unusable for K > 10 | High | High | Replace with `PosteriorFactorCopulaLoader` using amplitude encoding; do NOT use `NormalDistribution` for large K | Quantum reviewer |
| **R3** | SBI NPE convergence fails on heavy-tailed priors (Student-t, vine copulas) | Medium | High | Fallback to NLE or flow matching; monitor ESS during training; set early stopping at 50k steps if ESS < 100 | ML-workflow |
| **R4** | LUMI JAX-ROCm installation issues (JAX not fully supported on MI250X) | Medium | High | Use CPU-only NumPy/Xarray for MC; reserve GPU for Qiskit Aer only; test JAX on LUMI early | ML-ops |
| **R5** | Finnish apartment loan real data access blocked (StatFin/Eurostat licensing) | Low | High | Use synthetic data as fallback from Day 1; document data lineage for real data attempt | Data steward |
| **R6** | Vine copula structure selection combinatorial explosion | Medium | Medium | Constrain to D-vine sequential structure; limit pair-copula families to Gaussian, t, Clayton | Quantitative risk |
| **R7** | QSVT angle convention sensitivity (oblivious vs non-oblivious) causes systematic error | Medium | Medium | Test both conventions in Phase 4; use most robust for production; compare against classical ground truth | Quantum reviewer |
| **R8** | T-count scaling exceeds fault-tolerant thresholds for near-term hardware | High | High | Use early-version circuits (low degree) for validation; document resource requirements; assess IBM Quantum feasibility separately | Quantum reviewer |
| **R9** | PosteriorFactorCopulaLoader circuit depth too large for Aer simulation | Medium | Medium | Use qubit reduction techniques (basis encoding, amplitude encoding truncation); limit K to 10 for initial validation | Quantum reviewer |
| **R10** | Memory bottleneck: storing posterior samples + loss scenarios exceeds LUMI RAM | Low | High | Always aggregate on-the-fly; never store full scenario arrays; use streaming quantile algorithm | ML-ops |

---

## 8. Open Questions for the User

### Q1 (Critical-Path): Synthetic data only, or attempt real Finnish loan data fetch?

**Context**: Phase 1 requires data. Real data from StatFin/Eurostat requires API integration and licensing review. Synthetic data is guaranteed to work but lacks real-world fidelity.

**Options**:
- **A**: Synthetic only (low risk, fast start)
- **B**: Synthetic first, real data in parallel as Phase 1 extension
- **C**: Real data attempt first; fallback to synthetic after 1 week

**Recommendation**: Option B — start synthetic, attempt real data in parallel. Flag as critical-path if real data is required for publication.

---

### Q2 (Critical-Path): What is the maximum portfolio size K for quantum simulation?

**Context**: `PosteriorFactorCopulaLoader` must know K to design the circuit. Qiskit Aer simulation depth scales with K. The existing `multivariateGCI` used `NormalDistribution` which scales as 2^n qubits — unusable for K > 10.

**Options**:
- **A**: K = 10 (small, tractable, publication-quality comparison with classical possible)
- **B**: K = 50–100 (medium, more realistic, borderline for simulation)
- **C**: K = 100–1000 (large, but may exceed quantum resources)

**Recommendation**: Start with K = 10 for Phase 4 validation, then scale to K = 50–100 if resources permit. Document scaling curves in Phase 6.

---

### Q3: What is the minimum posterior sample count N for uncertainty propagation?

**Context**: Aggregating across posterior samples gives uncertainty bands. N = 1000–10000 is typical for SBI, but each sample requires a full QSVT circuit.

**Options**:
- **A**: N = 100 (fast, but wide uncertainty bands)
- **B**: N = 1000 (standard, good band width)
- **C**: N = 10000 (comprehensive, but 10x compute)

**Recommendation**: N = 1000 for initial experiments; N = 500 for Phase 4 degree sweep to save GPU hours.

---

### Q4: Which stress regimes should be the primary focus?

**Context**: Five regimes are defined (baseline, housing crash, rate shock, unemployment, liquidity). Training on all five is expensive. Focus on 2–3 may be sufficient.

**Options**:
- **A**: Baseline + Housing crash + Rate shock (most relevant for Finnish market)
- **B**: Baseline + Unemployment + Liquidity (macro-driven)
- **C**: All 5 (comprehensive but expensive)

**Recommendation**: Option A — housing and rate are most relevant for Finnish apartment loans. Add unemployment as third.

---

### Q5: Is Qiskit Aer GPU simulation required for Phase 4, or is Qiskit Aer CPU sufficient?

**Context**: Phase 4 GPU-hours estimate (30–80) assumes Qiskit Aer GPU simulation. CPU simulation would take 10x longer but may be sufficient if degree sweep is reduced.

**Options**:
- **A**: GPU (fast, but requires ROCm-compatible Qiskit Aer on LUMI)
- **B**: CPU only (slower, but more reliable availability)
- **C**: Hybrid (GPU for degrees ≤ 256, CPU for > 256)

**Recommendation**: Option C — test GPU first; fall back to CPU for high-degree circuits. Verify Qiskit Aer ROCm support on LUMI in Week 1.

---

## Appendix A: Module Directory Structure

```
QSVT4CRA/
├── Code/                          # Existing code (DO NOT rewrite)
│   ├── QSVT.py                    # QSVT circuit class (pyqsp 0.2.0)
│   ├── circuitsCRA.py             # get_expected_probability_circuit
│   ├── multivariateGCI.py        # MultivariateGCI_Poly, MultivariateGCI_Linear (TO BE REPLACED)
│   ├── AmplitudeLoading.py        # AmplitudeLoading, AmplitudeLoadingVar
│   └── utils.py                   # mapping(), bisection_search()
├── data/
│   ├── loader.py                  # NEW: real Finnish data loader
│   ├── synthetic.py               # NEW: synthetic portfolio generator
│   └── stress_regimes.py          # NEW: parametric stress regime definitions
├── simulator/
│   └── forward.py                 # NEW: JAX forward model (θ → x)
├── sbi/
│   ├── posterior.py               # NEW: NPE/NLE training
│   └── utils.py                   # NEW: prior spec, proposal, W&B logging
├── copula/
│   ├── gaussian.py                # NEW: Gaussian factor copula
│   ├── student_t.py               # NEW: Student-t factor copula
│   ├── vine.py                    # NEW: D-vine copula
│   └── low_rank.py                # NEW: low-rank factor copula
├── loader/
│   ├── posterior_factor_copula.py # NEW: PosteriorFactorCopulaLoader (replaces MultivariateGCI)
│   └── amplitude_loader.py        # NEW: optimized amplitude loading for large K
├── qsvt/
│   ├── approximator.py            # NEW: QSVT phase sequence computation (pyqsp)
│   ├── circuit.py                 # NEW: QSVTRiskCircuit (full circuit)
│   └── threshold.py               # NEW: threshold function polynomial construction
├── metrics/
│   ├── var_cvar.py                # NEW: classical VaR/CVaR computation
│   ├── quantum_error.py           # NEW: quantum vs classical error analysis
│   └── coverage.py                # NEW: SBC posterior coverage check
├── experiments/
│   ├── sbi_train.py              # CLI: SBI training entry
│   ├── mc_ground_truth.py         # CLI: MC ground truth generation
│   ├── qsvt_sweep.py              # CLI: QSVT degree sweep
│   ├── ood_robustness.py          # CLI: OOD robustness test
│   ├── resource_scaling.py        # CLI: resource scaling study
│   └── make_figures.py            # CLI: figure generation
├── configs/
│   ├── default.yaml               # Default config (Hydra)
│   ├── sbi_train.yaml             # SBI training config
│   ├── stress_regimes.yaml        # Stress regime definitions
│   └── quantum.yaml               # Quantum circuit config
├── docs/
│   ├── architecture.md            # THIS FILE
│   ├── linear_backlog.md          # Phase issues and milestones
│   └── hardware_notes.md          # LUMI hardware notes
├── Memory/
│   └── Projects/
│       ├── quantumhack.md         # BlockEncoder reference
│       └── trust-crisis-sbi.md    # vine copula + SBI reference
└── requirements.txt               # (existing)
```

---

## Appendix B: Interface Compatibility Checklist

For `PosteriorFactorCopulaLoader` to replace `MultivariateGCI_*`:

- [ ] `.num_qubits` attribute exists and returns int
- [ ] `.to_gate()` method exists and returns `qiskit.circuit.library.BlueprintGate`
- [ ] Constructor signature: `(θ: float32[D], K: int, max_loss: float32) → QuantumCircuit`
- [ ] Compatible with `get_expected_probability_circuit(uncertainty_model=loader, ...)` call in `circuitsCRA.py`
- [ ] Circuit depth < 10,000 (for Aer simulation tractability)
- [ ] Circuit width (qubits) < 100 (for memory tractability)

---

## Appendix C: Reference Notes

- **pyqsp 0.2.0**: Pinned in `requirements.txt`. Known degree limits (< 64–128). See `Code/QSVT.py` line 30–32 for usage.
- **Qiskit 1.1.2**: Current version. `qiskit-finance 0.4.1` provides `NormalDistribution` (do NOT use for large K).
- **sbi 0.22.0**: Not in requirements.txt — add explicitly. Required for NPE/NLE training.
- **LUMI MI250X**: ROCm support required for JAX on GPU. Verify before Phase 1.
- **BlockEncoder** (`Memory/Projects/quantumhack.md`): Reference implementation for efficient amplitude loading.

---

*Created by: dev-team-planner | Review by: ML-workflow, Quantum computing, Quantitative risk*