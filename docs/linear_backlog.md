# QSVT4CRA Research Run — Linear Backlog

**Project**: Posterior-Propagated Factor-Copula QSVT for Apartment Loan Portfolio Risk  
**Linear Project**: `project_465003017` (LUMI)  
**Compute Budget**: 50–150 GPU-hours, <500 CPU-hours on LUMI-G (MI250X)  
**Date Created**: 2026-06-06

---

## Milestones

### M1: Foundation — SBI Posterior + Factor Copula
**Target Completion**: Week 2  
**Issues**: Phase 1, Phase 2  
**Goal**: Posterior distribution over dependence parameters + working factor-copula risk model

### M2: Ground Truth + QSVT Loader
**Target Completion**: Week 4  
**Issues**: Phase 3, Phase 4  
**Goal**: Classical Monte Carlo ground truth established; `PosteriorFactorCopulaLoader` exports U(θ) for quantum compilation

### M3: Shift + Scaling
**Target Completion**: Week 6  
**Issues**: Phase 5, Phase 6  
**Goal**: Distribution shift experiment complete; quantum resource scaling curves produced

### M4: Figures + Narrative
**Target Completion**: Week 8  
**Issues**: Phase 7  
**Goal**: 7 publication figures + 1 hackathon poster figure ready for submission

---

## Phase Issues

---

### Phase 1: Construct Posterior over Dependence Parameters via Robust SBI

**Issue ID**: BACKLOG-001  
**Title**: Phase 1 — SBI Posterior over Copula/Factor Loadings  
**Priority**: Urgent (blocks Phases 2–7)  
**RQ**: RQ1 — Can posterior-propagated SBI capture regime-varying dependence better than point-estimate GCI?  
**Labels**: `phase-1`, `sbi`, `posterior-inference`, `critical-path`

#### Description

Train Robust SBI (NSM-Bayes, NPE, NLE, or flow matching) to construct `p(θ | data)` where:

```
θ = {factor_loadings, default_thresholds, tail_dependence, copula_parameters}
```

- Use broad market regimes + 4 stress regimes (housing crash, rate shock, regional unemployment, liquidity crisis)
- Goal: full posterior `p(θ | data)`, not point estimate
- Must integrate with existing `multivariateGCI` data loaders

**Acceptance Criteria**:
- [ ] Posterior samples `p(θ | data)` for all 5 regime types
- [ ] Tail dependence and factor loadings show regime-dependent structure
- [ ] NPE/NLE convergence verified (ESS > 200 per parameter)
- [ ] Comparison: posterior spread vs fixed-GCI point estimate divergence
- [ ] Reproducible: seed, config, and training curves logged to W&B

**Dependencies**: None (foundational)  
**Blocked By**: None  
**Blocks**: Phase 2, Phase 5, Phase 7 (Fig 1–2)

**Compute Budget Estimate**:
- GPU-hours: 20–40 (NPE training on MI250X)
- CPU-hours: 10–20 (preprocessing, evaluation)
- Wall-clock: 8–16 hours (distributed training)

**LUMI Job Target**: `sbatch rsync_to_lumi.sh slurm_qsvt4cra_research.sh` — SBI training job

#### Sub-tasks

1. **1.1 — Data Loader Integration**
   - Connect Finnish housing/apartment loan data to SBI training pipeline
   - Acceptance: `DataLoader` yields `x, θ` pairs for all 5 regimes
   - Labels: `data-pipeline`

2. **1.2 — Prior Specification**
   - Define priors over `{factor_loadings, default_thresholds, tail_dependence, copula_params}`
   - Acceptance: Prior predictive checks pass (divergence < 0.5 nats from historical moments)
   - Labels: `bayesian-inference`

3. **1.3 — NPE/NLE/Flow-Matching Training**
   - Train SBI posteriors for each regime separately
   - Acceptance: Test log-likelihood improves > 2 nats over prior; ESS > 200
   - Labels: `sbi-training`

4. **1.4 — Regime-Pooled Posterior Validation**
   - Pool posteriors across regimes; verify posterior mixing
   - Acceptance: Pooled posterior captures regime-shifted dependence structure
   - Labels: `validation`

5. **1.5 — W&B Logging & Reproducibility**
   - Log all training curves, configs, seeds
   - Acceptance: Run fully reproducible from logged config
   - Labels: `ml-ops`

**Required Reviewer Gates**: ML-workflow reviewer (NPE convergence check), data steward (Finnish housing data usage)

---

### Phase 2: Factor-Copula Risk Model

**Issue ID**: BACKLOG-002  
**Title**: Phase 2 — Factor-Copula Risk Model Implementation  
**Priority**: Urgent (blocks Phase 3–4)  
**RQ**: RQ2 — Does factor-copula model preserve tail-dependence structure from posterior?  
**Labels**: `phase-2`, `factor-copula`, `risk-model`, `critical-path`

#### Description

Implement factor-copula risk model with hierarchy:

```
Baseline Gaussian → Student-t → Vine → D-vine → Low-rank factor copula
```

For each posterior sample `θ⁽ⁱ⁾`, generate `P(loss | θ⁽ⁱ⁾)`.

**Acceptance Criteria**:
- [ ] All 5 copula model variants implemented and tested
- [ ] Low-rank factor copula matches full-vine CDF within ε < 0.01 for 95th percentile
- [ ] Posterior samples correctly condition risk model (loss CDF shifts match posterior spread)
- [ ] Factor loadings interpretable: housing price sensitivity, rate sensitivity, unemployment sensitivity
- [ ] Unit tests pass for all copula variants

**Dependencies**: Phase 1 (posterior samples required)  
**Blocked By**: Phase 1  
**Blocks**: Phase 3, Phase 4, Phase 7 (Fig 3)

**Compute Budget Estimate**:
- GPU-hours: 5–10 (copula fitting, validation)
- CPU-hours: 20–40 (Monte Carlo loss generation per posterior sample)
- Wall-clock: 4–8 hours

**LUMI Job Target**: CPU job on LUMI-G for copula fitting

#### Sub-tasks

1. **2.1 — Gaussian & Student-t Copula Baseline**
   - Implement factor-copula with Gaussian and Student-t margins
   - Acceptance: CDF matches `scipy.stats` reference implementation
   - Labels: `copula-implementation`

2. **2.2 — Vine & D-vine Copula Extension**
   - Implement regular vine and D-vine structures
   - Acceptance: Vine structure selection via BIC; D-vine sequential.
   - Labels: `copula-implementation`

3. **2.3 — Low-Rank Factor Copula**
   - Implement low-rank approximation; verify CDF error< 0.01 at 95th
   - Acceptance: Low-rank matches full-vine within tolerance
   - Labels: `copula-implementation`

4. **2.4 — Posterior Conditioning**
   - Connect posterior samples `θ⁽ⁱ⁾` to risk model; generate `P(loss | θ⁽ⁱ⁾)`
   - Acceptance: Loss CDF shift matches posterior spread of dependence parameters
   - Labels: `risk-model`

5. **2.5 — Unit Tests & Reference Comparison**
   - Write unit tests for all copula variants
   - Acceptance: All tests pass; CDF error < 1e-3 vs reference
   - Labels: `testing`

**Required Reviewer Gates**: Quantitative risk reviewer (copula validity), ML-workflow reviewer (posterior conditioning)

---

### Phase 3: Classical Ground Truth via Monte Carlo

**Issue ID**: BACKLOG-003  
**Title**: Phase 3 — Monte Carlo Ground Truth (1e6–1e8 Scenarios)  
**Priority**: High (blocks Phase 4 validation)  
**RQ**: RQ3 — What is the true Loss CDF, VaR, CVaR at 95/99/99.9 confidence?  
**Labels**: `phase-3`, `monte-carlo`, `ground-truth`, `critical-path`

#### Description

Compute classical ground truth using 1e6–1e8 Monte Carlo scenarios per experiment.

**Acceptance Criteria**:
- [ ] Loss CDF computed for all5 regimes at 1e6 scenario minimum
- [ ] VaR₉₅, VaR₉₉, VaR₉₉.₉ computed with confidence intervals
- [ ] CVaR₉₅, CVaR₉₉, CVaR₉₉.₉ computed
- [ ] Tail probability (P(loss > threshold)) computed
- [ ] Regime comparison table: baseline vs stress regimes
- [ ] Runtime < 2 hours per experiment on LUMI-G

**Dependencies**: Phase 2 (factor-copula model required)  
**Blocked By**: Phase 2  
**Blocks**: Phase 4, Phase 7 (Fig 4)

**Compute Budget Estimate**:
- GPU-hours: 0 (CPU-only MC)
- CPU-hours: 200–400 (1e6–1e8 scenarios ×5 regimes)
- Wall-clock: 1–2 hours per experiment (vectorized NumPy/Xarray)

**LUMI Job Target**: CPU job on LUMI-G — Monte Carlo ground truth

#### Sub-tasks

1. **3.1 — Monte Carlo Engine**
   - Vectorized MC generator using NumPy/Xarray
   - Acceptance: 1e8 scenarios generated in < 2 hours
   - Labels: `monte-carlo`

2. **3.2 — VaR/CVaR Computation**
   - Compute all VaR and CVaR metrics with bootstrapped CIs
   - Acceptance: CI width < 0.005 for VaR₉₅
   - Labels: `risk-metrics`

3. **3.3 — Regime Comparison**
   - Generate comparison table across all 5 regimes
   - Acceptance: Table shows regime-shift in tail risk
   - Labels: `analysis`

4. **3.4 — Validation Against Known Baselines**
   - Validate MC ground truth against known analytical results (Gaussian case)
   - Acceptance: Error < 1e-3 vs analytical CDF
   - Labels: `validation`

**Required Reviewer Gates**: Quantitative risk reviewer (VaR/CVaR methodology), ML-workflow reviewer (CI methodology)

---

### Phase 4: PosteriorFactorCopulaLoader + QSVT Compilation

**Issue ID**: BACKLOG-004  
**Title**: Phase 4 — QSVT Loader& Polynomial Approximation Sweep  
**Priority**: High (blocks Phase 5–6)  
**RQ**: RQ2 (continued) — Can QSVT approximate the posterior-propagated loss CDF accurately?  
**Labels**: `phase-4`, `qs `quantum`, `amplitude-loading`, `critical-path`

#### Description

Replace `multivariateGCI` loader with `PosteriorFactorCopulaLoader` that exports `U(θ)` for quantum compilation. QSVT polynomial approximation sweep over degrees `[16, 32, 64, 128, 256, 512, 1024]` with multiple angle conventions and threshold mappings.

**Acceptance Criteria**:
- [ ] `PosteriorFactorCopulaLoader` implemented and tested
- [ ] Exports `U(θ)` amplitude loading circuit for Qiskit
- [ ] QSVT degree sweep: [16, 32, 64, 128, 256, 512, 1024] completed
- [ ] Multiple angle conventions tested (oblivious, non-oblivious)
- [ ] Threshold mappings validated
- [ ] CDF, VaR, CVaR, tail-probability error measured vs classical ground truth
- [ ] pyqsp 0.2.0 degree limits respected (max ~1024)

**Dependencies**: Phase 1 (posterior), Phase 2 (factor-copula), Phase 3 (ground truth)  
**Blocked By**: Phase 1, Phase 2, Phase 3  
**Blocks**: Phase 5, Phase 6, Phase 7 (Fig 5–6)

**Compute Budget Estimate**:
- GPU-hours: 30–80 (Qiskit transpilation, QSVT circuit compilation)
- CPU-hours: 50–100 (pyqsp polynomial fitting, circuit simulation)
- Wall-clock: 4–12 hours per degree (circuit compilation + simulation)

**LUMI Job Target**: GPU job on LUMI-G — Qiskit Aer simulation + QSVT compilation

#### Sub-tasks

1. **4.1 — PosteriorFactorCopulaLoader Implementation**
   - Load posterior samples; export `U(θ)` amplitude loading
   - Acceptance: Loader produces valid Qiskit `QuantumCircuit`
   - Labels: `quantum-loader`

2. **4.2 — QSVT Polynomial Fitting (pyqsp)**
   - Fit QSVT polynomials for degrees [16, 32, 64, 128, 256, 512, 1024]
   - Acceptance: Polynomial approximation error < 1e-3 in L∞ norm
   - Labels: `qs `quantum`

3. **4.3 — Angle Convention Testing**
   - Test oblivious vs non-oblivious angle conventions
   - Acceptance: Both conventions produce valid CDF approximations
   - Labels: `qs `quantum`

4. **4.4 — Threshold Mapping Validation**
   - Map loss thresholds to QSVT angle parameters
   - Acceptance: Threshold mapping error < 1e-2
   - Labels: `qs `quantum`

5. **4.5 — Error Metrics vs Classical Ground Truth**
   - Measure CDF error, VaR error, CVaR error, tail probability error
   - Acceptance: Error metrics logged per degree and regime
   - Labels: `validation`

6. **4.6 — Qiskit Aer Simulation**
   - Simulate QSVT circuits for small degrees (≤128) on Qiskit Aer
   - Acceptance: Simulation results match pyqsp theoretical predictions
   - Labels: `quantum-simulation`

**Required Reviewer Gates**: Quantum computing reviewer (QSVT correctness), ML-workflow reviewer (error metric methodology)

---

### Phase 5: Distribution Shift Experiment

**Issue ID**: BACKLOG-005  
**Title**: Phase 5 — Distribution Shift: Train on Regime A, Test on Regime B  
**Priority**: High (tests core hypothesis)  
**RQ**: RQ4 — Does posterior-aware approach degrade more gracefully under distribution shift?  
**Labels**: `phase-5`, `distribution-shift`, `robustness`, `hypothesis-test`

#### Description

Train on Regime A, test on Regime B (low-rate, high-rate, housing crash). Compare point-estimate GCI vs posterior-propagated factor copula.

**Acceptance Criteria**:
- [ ] Train/val split: Regime A (broad market) → Regimes B (stress scenarios)
- [ ] Point-estimate GCI baseline: VaR error reported for each shift
- [ ] Posterior-propagated approach: VaR error reported for each shift
- [ ] Hypothesis validated: posterior-aware degrades more gracefully (lower VaR error on unseen regimes)
- [ ] Statistical significance: p-value < 0.05 for improvement
- [ ] Visualization: shift degradation curves for both approaches

**Dependencies**: Phase 1 (posterior), Phase 2 (factor-copula), Phase 3 (ground truth), Phase 4 (QSVT loader)  
**Blocked By**: Phase 4  
**Blocks**: Phase 7 (Fig 7)

**Compute Budget Estimate**:
- GPU-hours: 10–20 (re-training SBI on Regime A subset)
- CPU-hours: 20–40 (shift evaluation)
- Wall-clock: 4–8 hours

**LUMI Job Target**: GPU job on LUMI-G — shift experiment

#### Sub-tasks

1. **5.1 — Train/Val Regime Split**
   - Define Regime A (broad market) and Regime B (stress scenarios)
   - Acceptance: Split documented; no data leakage
   - Labels: `experiment-design`

2. **5.2 — Point-Estimate GCI Baseline**
   - Run fixed-parameter GCI on Regime B after training on Regime A
   - Acceptance: VaR errors logged for all Regime B variants
   - Labels: `baseline`

3. **5.3 — Posterior-Propagated Factor Copula**
   - Run posterior-propagated approach on Regime B
   - Acceptance: VaR errors logged; posterior spread accounted for
   - Labels: `experiment`

4. **5.4 — Statistical Significance Testing**
   - Compute p-values for improvement over baseline
   - Acceptance: p-value < 0.05 for primary comparison
   - Labels: `statistics`

5. **5.5 — Visualization**
   - Generate shift degradation curves
   - Acceptance: Publication-quality figure generated
   - Labels: `visualization`

**Required Reviewer Gates**: ML-workflow reviewer (statistical methodology), quantitative risk reviewer (shift interpretation)

---

### Phase 6: Quantum Resource Scaling

**Issue ID**: BACKLOG-006  
**Title**: Phase 6 — Quantum Resource Scaling Curves  
**Priority**: Medium (posters& paper appendix)  
**RQ**: RQ2 (scaling aspect) — How do quantum resources scale with portfolio size?  
**Labels**: `phase-6`, `scaling`, `resource-estimation`

#### Description

For portfolio sizes `[10, 50, 100, 500, 1000]` loans, estimate qubits, QSVT degree, T-count, depth. Produce scaling curves.

**Acceptance Criteria**:
- [ ] Resource estimates for all 5 portfolio sizes
- [ ] Scaling curves: qubits vs loans, T-count vs loans, depth vs loans
- [ ] Comparison: QSVT degree required vs classical MC cost
- [ ] Feasibility assessment: which portfolio sizes are tractable on near-term hardware
- [ ] Resource estimation methodology documented

**Dependencies**: Phase 4 (QSVT loader and degree sweep)  
**Blocked By**: Phase 4  
**Blocks**: Phase 7 (Fig 6 scaling panel)

**Compute Budget Estimate**:
- GPU-hours: 5–10 (resource estimation runs)
- CPU-hours: 10–20 (estimation, curve fitting)
- Wall-clock: 2–4 hours

**LUMI Job Target**: CPU job on LUMI-G — resource estimation

#### Sub-tasks

1. **6.1 — Qubit Count Estimation**
   - Estimate qubit requirements per portfolio size
   - Acceptance: Estimates reported with methodology documented
   - Labels: `resource-estimation`

2. **6.2 — T-Count & Depth Estimation**
   - Estimate T-count and circuit depth for each portfolio size
   - Acceptance: Estimates reported with assumptions documented
   - Labels: `resource-estimation`

3. **6.3 — Scaling Curve Fitting**
   - Fit scaling curves; identify crossover points
   - Acceptance: Scaling exponents reported
   - Labels: `analysis`

4. **6.4 — Feasibility Assessment**
   - Assess tractability on near-term hardware (IBM Quantum, IonQ)
   - Acceptance: Feasibility table generated
   - Labels: `analysis`

**Required Reviewer Gates**: Quantum computing reviewer (resource methodology), ML-workflow reviewer (scaling fit)

---

### Phase 7: Publication Figures + Hackathon Poster

**Issue ID**: BACKLOG-007  
**Title**: Phase 7 — 7 Publication Figures + Hackathon Poster Figure  
**Priority**: Medium (end-stage deliverable)  
**RQ**: All (synthesis) — End-to-end pipeline demonstration  
**Labels**: `phase-7`, `figures`, `publication`, `poster`

#### Description

Generate all 7 publication figures + 1 hackathon poster figure (Fig 7 = end-to-end pipeline).

**Figure Plan**:
- **Fig 1**: Posterior over dependence parameters (from Phase 1)
- **Fig 2**: Factor-copula risk model schematic (from Phase 2)
- **Fig 3**: Loss CDF comparison: Gaussian vs Student-t vs Vine vs D-vine vs Low-rank (from Phase 2)
- **Fig 4**: Classical ground truth: VaR/CVaR by regime (from Phase 3)
- **Fig 5**: QSVT approximation error vs degree (from Phase 4)
- **Fig 6**: Quantum resource scaling curves (from Phase 6)
- **Fig 7**: End-to-end pipeline diagram (hackathon poster figure)
- **Fig 8**: Distribution shift comparison (from Phase 5)

**Acceptance Criteria**:
- [ ] All 7 publication figures generated at publication quality (300 DPI, LaTeX-compatible)
- [ ] Hackathon poster figure (Fig 7) generated
- [ ] Figure captions written
- [ ] Source code for all figures committed and reproducible
- [ ] Figure data logged to W&B

**Dependencies**: Phase 1–6 (all prior phases)  
**Blocked By**: Phase 5, Phase 6  
**Blocks**: None (final deliverable)

**Compute Budget Estimate**:
- GPU-hours: 5–10 (figure rendering)
- CPU-hours: 10–20 (plotting, caption writing)
- Wall-clock: 4–8 hours

**LUMI Job Target**: CPU job on LUMI-G — figure generation

#### Sub-tasks

1. **7.1 — Figures 1–2 (Posterior + Factor Copula)**
   - Generate Phase 1 and Phase 2 figures
   - Acceptance: Publication quality; captions written
   - Labels: `figures`

2. **7.2 — Figure 3 (Loss CDF Comparison)**
   - Generate CDF comparison across copula variants
   - Acceptance: Publication quality; captions written
   - Labels: `figures`

3. **7.3 — Figure 4 (Classical Ground Truth)**
   - Generate VaR/CVaR regime comparison
   - Acceptance: Publication quality; captions written
   - Labels: `figures`

4. **7.4 — Figure 5 (QSVT Approximation Error)**
   - Generate error vs degree curves
   - Acceptance: Publication quality; captions written
   - Labels: `figures`

5. **7.5 — Figure 6 (Resource Scaling)**
   - Generate scaling curves
   - Acceptance: Publication quality; captions written
   - Labels: `figures`

6. **7.6 — Figure 7 (End-to-End Pipeline)**
   - Generate hackathon poster figure
   - Acceptance: Poster quality; pipeline steps clearly shown
   - Labels: `figures`

7. **7.7 — Figure8 (Distribution Shift)**
   - Generate shift comparison figure
   - Acceptance: Publication quality; captions written
   - Labels: `figures`

8. **7.8 — W&B Logging & Reproducibility**
   - Log all figure data and source code
   - Acceptance: All figures reproducible from logged artifacts
   - Labels: `ml-ops`

**Required Reviewer Gates**: All phase reviewers (figure accuracy check), PI (narrative approval)

---

## Dependency Graph

```
Phase 1 (SBI Posterior)
    │
    ├─────────────────────┬─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼                     ▼
Phase 2               Phase 5               Phase 7               Phase 7
(Factor Copula)    (Distribution Shift)    (Fig 1-2)            (Fig 8 - dep on Phase 5)
    │                     │                     │                     │
    ▼                     │                     │                     │
Phase 3                  │                     │                     │
(Ground Truth)          │                     │                     │
    │                     │                     │                     │
    └─────────┬───────────┘                     │                     │
              ▼                                 │                     │
        Phase 4                                 │                     │
  (QSVT Loader)                                 │                     │
        │                                      │                     │
        ├──────────────┬───────────────────────┘                     │
        ▼             ▼                                             │
    Phase 5       Phase 6                                           │
(Shift Exp)    (Scaling)                                             │
        │             │                                             │
        └──────┬──────┘                                             │
               ▼                                                    │
           Phase 7 ─────────────────────────────────────────────────┘
        (All Figures)
```

**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 7

---

## Risk Register

| Risk ID | Description | Likelihood | Impact | Mitigation | Owner |
|---------|-------------|------------|--------|------------|-------|
| R1 | pyqsp 0.2.0 degree limits (< 1024) restrict high-accuracy approximations | High | Medium | Pre-compute optimal degree per accuracy target; use piecewise approximation | ML-workflow |
| R2 | Qiskit `NormalDistribution` qubit scaling (2^n qubits for n assets) | High | High | Use amplitude loading with qubit-efficient encodings (e.g., basis encoding) | Quantum reviewer |
| R3 | SBI0.22.0 NPE convergence on heavy-tailed priors (Student-t, vine copulas) | Medium | High | Fallback to NLE or flow matching; monitor ESS during training | ML-workflow |
| R4 | LUMI JAX-ROCm availability (JAX not fully supported on MI250X) | Medium | High | Use CPU-only NumPy/Xarray for MC; reserve GPU for Qiskit Aer if ROCm available | ML-ops |
| R5 | Finnish housing data access/licensing restrictions | Low | High | Use synthetic data as fallback; document data lineage | Data steward |
| R6 | Vine copula structure selection (combinatorial explosion) | Medium | Medium | Use D-vine sequential structure; constrain to pair-copula families | Quantitative risk |
| R7 | QSVT angle convention sensitivity (oblivious vs non-oblivious) | Medium | Medium | Test both conventions; use most robust for production | Quantum reviewer |
| R8 | T-count scaling exceeds fault-tolerant thresholds | High | High | Use early-version circuits for validation; document resource requirements | Quantum reviewer |

---

## Effort Estimates Summary

| Phase | GPU-hours | CPU-hours | Wall-clock | LUMI Job Target |
|-------|-----------|-----------|------------|-----------------|
| Phase 1 | 20–40 | 10–20 | 8–16 hrs | GPU (SBI training) |
| Phase 2 | 5–10 | 20–40 | 4–8 hrs | CPU (copula fitting) |
| Phase 3 | 0 | 200–400 | 1–2 hrs | CPU (MC ground truth) |
| Phase 4 | 30–80 | 50–100 | 4–12 hrs | GPU (Qiskit + QSVT) |
| Phase 5 | 10–20 | 20–40 | 4–8 hrs | GPU (shift exp) |
| Phase 6 | 5–10 | 10–20 | 2–4 hrs | CPU (estimation) |
| Phase 7 | 5–10 | 10–20 | 4–8 hrs | CPU (figures) |
| **Total** | **75–170** | **320–640** | **27–54 hrs** | — |

**Note**: Budget target is 50–150 GPU-hours and <500 CPU-hours. Phase 4 (QSVT compilation) is the GPU bottleneck. Consider reducing QSVT degree sweep to [16, 64, 256, 1024] if GPU budget constrained.

---

## Top 3 Critical-Path Items (Flag to User)

### 1. 🔴 Phase 1 — SBI Posterior Training (BLOCKS EVERYTHING)
**Why Critical**: Phase 1 produces the posterior samples that all subsequent phases depend on. Any delay in NPE convergence, data access, or prior specification will cascade.

**Immediate Action Required**:
- Confirm Finnish housing data access or establish synthetic fallback
- Verify SBI0.22.0 installation on LUMI
- Set up W&B project for experiment tracking

**Estimated Impact**: 2-week delay if blocked

### 2. 🔴 Phase 4 — QSVT Loader + Degree Sweep (GPU BOTTLENECK)
**Why Critical**: Phase 4 is the GPU-intensive core of the project (30–80 GPU-hours). pyqsp degree limits and Qiskit scaling are known risks that could require redesign.

**Immediate Action Required**:
- Verify pyqsp 0.2.0 compatibility with desired degrees
- Test Qiskit `NormalDistribution` qubit scaling for 1000-loan portfolio
- Establish amplitude loading strategy before Phase 4 start

**Estimated Impact**: 1-week delay if degree limits hit

### 3. 🟡 Phase 3 — Monte Carlo Ground Truth (ACCURACY GATE)
**Why Critical**: Phase 3 establishes the ground truth that validates all QSVT approximations. Insufficient scenario count or incorrect VaR/CVaR methodology would invalidate Phase 4–5 results.

**Immediate Action Required**:
- Confirm NumPy/Xarray vectorized MC can reach 1e8 scenarios in < 2 hours
- Validate VaR/CVaR confidence interval methodology with quantitative risk reviewer
- Establish regime comparison table format early

**Estimated Impact**: 3-day delay if vectorization insufficient

---

## Reviewer Gates Summary

| Phase | Primary Reviewer | Secondary Reviewer | Gate Type |
|-------|-----------------|-------------------|-----------|
| Phase 1 | ML-workflow | Data steward | NPE convergence + data lineage |
| Phase 2 | Quantitative risk | ML-workflow | Copula validity + posterior conditioning |
| Phase 3 | Quantitative risk | ML-workflow | VaR/CVaR methodology + CI |
| Phase 4 | Quantum computing | ML-workflow | QSVT correctness + error metrics |
| Phase 5 | ML-workflow | Quantitative risk | Statistical methodology + shift interpretation |
| Phase 6 | Quantum computing | ML-workflow | Resource methodology + scaling fit |
| Phase 7 | All phase reviewers | PI | Figure accuracy + narrative approval |

---

## Notes for Orchestrator

1. **Linear MCP tools not detected** — this backlog is written to `docs/linear_backlog.md`. If Linear MCP becomes available, import these issues into `project_465003017`.

2. **Phase ordering**: Phase 1 → Phase 2 → Phase 3 → Phase 4 is the critical path. Phases 5 and 6 can run in parallel with Phase 4 once data dependencies are met.

3. **GPU budget allocation**: Phase 4 (QSVT) should get priority allocation. Phase 1 (SBI) is second priority. Phases 5–6 share remaining budget.

4. **Data gaps flagged**: VaR/CVaR computation, stress regimes, and Finnish housing data are confirmed gaps. Phase 1 and Phase 3 sub-tasks address these directly.

5. **Known technical risks**: pyqsp degree limits (R1), Qiskit qubit scaling (R2), and JAX-ROCm availability (R4) are the top risks. Mitigation strategies are documented in the risk register.

---

*Created by: dev-team-pm | Source: QSVT4CRA Research Run Brief*
