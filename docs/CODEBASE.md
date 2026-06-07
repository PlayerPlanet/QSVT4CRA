# QSVT4CRA Codebase Reference

> Generated: 2026-06-07. This is the canonical tour of the QSVT4CRA
> source tree, intended for new contributors, reviewers, and
> paper-writing collaborators.

The repository implements the methodology of the paper
**"Implementing Credit Risk Analysis with Quantum Singular Value
Transformation"**. The high-level architecture is described in
[`architecture.md`](architecture.md); this document maps every source
file to its role in that architecture.

## 1. Bird's-eye view

```
QSVT4CRA/
├── Code/                    # Original QSVT-on-GCI prototype (K≤10)
│   ├── QSVT.py              #   core QSVT circuit class
│   ├── circuitsCRA.py       #   get_expected_probability_circuit
│   ├── AmplitudeLoading.py  #   amplitude-encoding primitives
│   ├── multivariateGCI.py   #   MultivariateGCI_Poly / _Linear
│   ├── utils.py             #   mapping() + bisection_search()
│   ├── generate_dataset.py  #   Statistics Finland PxWeb fetch
│   ├── dataset_regions.py   #   generated K=17 region data
│   └── DATASET.md           #   dataset documentation
│
├── copula/                  # Factor-copula simulators
│   ├── base.py              #   abstract FactorCopula
│   ├── gaussian.py          #   one-factor Gaussian copula
│   ├── student_t.py         #   one-factor Student-t copula
│   ├── vine.py              #   hand-rolled D-vine copula
│   └── low_rank.py          #   low-rank factor copula
│
├── data/                    # Data layer
│   ├── synthetic.py         #   SyntheticPortfolioGenerator
│   ├── stress_regimes.py    #   StressRegimeGenerator + REGIME_SPECS
│   └── loader.py            #   RealDataLoader (stub)
│
├── loader/                  # Quantum state-preparation loaders
│   ├── posterior_factor_copula.py  # PosteriorFactorCopulaLoader
│   └── amplitude_loader.py         # AmplitudeLoader
│
├── qsvt/                    # QSVT circuits and approximators
│   ├── circuit.py           #   QSVTRiskCircuit
│   ├── approximator.py      #   QSVTApproximator (pyqsp wrapper)
│   └── threshold.py         #   ThresholdFunction
│
├── metrics/                 # Classical risk metrics & MC ground truth
│   ├── var_cvar.py          #   VaR / CVaR / loss CDF
│   ├── ground_truth.py      #   GroundTruthMC
│   └── quantum_error.py     #   quantum-vs-classical error
│
├── simulator/               # Forward simulators
│   └── forward.py           #   NumPy + JAX forward models
│
├── sbi_pipeline/            # Simulation-Based Inference
│   ├── posterior.py         #   NPE / NLE / flow-matching pipelines
│   └── utils.py             #   prior + W&B helpers
│
├── compiler_backend/        # Hardware-aware compilation
│   └── heron.py             #   IBM Heron r3 + pyzx post-opt
│
├── experiments/             # CLI entry points
│   ├── sbi_train.py         #   Phase 1: SBI training
│   ├── mc_ground_truth.py   #   Phase 3: MC ground truth
│   ├── qsvt_sweep.py        #   Phase 4: degree sweep
│   ├── boston_qpu.py        #   Phase 4: real IBM QPU run
│   ├── heron_simulation.py  #   Phase 4: Aer-on-Heron sim
│   ├── analyze_degree_sweep.py  # Phase 4: post-process results
│   ├── check_qpu_jobs.py    #   job-status checker
│   ├── ood_robustness.py    #   Phase 5: distribution shift
│   ├── resource_scaling.py  #   Phase 6: resource estimator
│   ├── make_figures.py      #   Phase 7: 8 publication figures
│   └── sbi_train.py         #   Phase 1: SBI training
│
├── tests/                   # pytest suite
│
├── lumi_deployment/         # LUMI-G Slurm infrastructure
│   ├── setup_lumi_env.sh
│   ├── dispatcher.sh
│   ├── slurm_*.sh
│   ├── profiles/            # 17 Slurm profile presets
│   ├── dispatch_all.sh
│   ├── rsync_to_lumi.sh
│   └── submit_from_windows.ps1
│
├── scale_runner.py          # Multi-node launcher abstraction
├── CRA_QSVT.ipynb           # Primary demo notebook
├── requirements.txt         # Pinned dependencies
└── docs/                    # You are here
```

## 2. Documentation map

This directory contains the following reference documents:

| File | Topic |
|---|---|
| [`architecture.md`](architecture.md) | System architecture, design decisions, compute budget, risk register (pre-existing) |
| [`linear_backlog.md`](linear_backlog.md) | Phased backlog, milestones, prior research notes (pre-existing) |
| [`hardware_notes.md`](hardware_notes.md) | LUMI module stack, known gotchas (pre-existing) |
| [`lumi_submission_guide.md`](lumi_submission_guide.md) | Step-by-step LUMI job submission (pre-existing) |
| [`lumi_scaling_strategy.md`](lumi_scaling_strategy.md) | What/why of the scaling matrix (pre-existing) |
| [`lumi_session_2026-06-06.md`](lumi_session_2026-06-06.md) | Session log (pre-existing) |
| [`phase1_ml_gate.md`](phase1_ml_gate.md) | Phase 1 ML gate criteria (pre-existing) |
| [`phase1_review.md`](phase1_review.md) | Phase 1 review notes (pre-existing) |
| [`prior_art.md`](prior_art.md) | Prior-art survey (pre-existing) |
| [`CODEBASE.md`](CODEBASE.md) | **This document — code-level tour of the entire repository** |
| [`code_documentation.md`](code_documentation.md) | `Code/` (original QSVT-on-GCI prototype) |
| [`copula_documentation.md`](copula_documentation.md) | `copula/` (factor-copula simulators) |
| [`data_documentation.md`](data_documentation.md) | `data/` (synthetic, stress, real loader) |
| [`loader_documentation.md`](loader_documentation.md) | `loader/` (state-preparation loaders) |
| [`qsvt_documentation.md`](qsvt_documentation.md) | `qsvt/` (QSVT circuits and approximators) |
| [`metrics_documentation.md`](metrics_documentation.md) | `metrics/` (VaR / CVaR / MC ground truth / quantum error) |
| [`simulator_documentation.md`](simulator_documentation.md) | `simulator/` (forward simulators) |
| [`sbi_pipeline_documentation.md`](sbi_pipeline_documentation.md) | `sbi_pipeline/` (SBI training) |
| [`compiler_backend_documentation.md`](compiler_backend_documentation.md) | `compiler_backend/` (IBM Heron compiler) |
| [`experiments_documentation.md`](experiments_documentation.md) | `experiments/` (CLI entry points) |
| [`lumi_deployment_documentation.md`](lumi_deployment_documentation.md) | `lumi_deployment/` (Slurm infrastructure) |
| [`tests_documentation.md`](tests_documentation.md) | `tests/` (pytest suite) |
| [`scale_runner_documentation.md`](scale_runner_documentation.md) | `scale_runner.py` (multi-node launcher) |

## 3. End-to-end data flow (recap)

```
Data (real or synthetic)
  └── theta_baseline     (parameter vector, dim 2K+4)
        │
        ▼
StressRegimeGenerator.apply(theta_baseline, regime, shock)
        │
        ▼  θ_perturbed
ForwardSimulator (NumPy or JAX).simulate(θ_perturbed, n_scenarios)
        │
        ▼  (θ_i, x_i) pairs
SBI pipeline.train(pairs)        ←── posterior_nn (NPE), NSF (NLE),
        │                                or ConditionalMAF (FM)
        ▼  posterior
Posterior.sample((N,))           ←── N posterior samples
        │
        ▼  {θ⁽ⁱ⁾}
Per-sample factor-copula MC     ←── GaussianFactorCopula / StudentTFactorCopula
        │
        ▼  {L_j}
PosteriorFactorCopulaLoader     ←── amplitude encoding of loss distribution
        │
        ▼  QuantumCircuit
QSVT phase approximation        ←── ThresholdFunction → pyqsp
        │
        ▼  QSVT(QSVT) over U
QSVTRiskCircuit / get_expected_probability_circuit
        │
        ▼  Qiskit QuantumCircuit
Compiler backend                ←── Heron r3 calibration-aware pass manager
        │                                (+ optional pyzx round-trip)
        ▼  ISA circuit
IBM QPU or Qiskit Aer           ←── execute and measure
        │
        ▼  counts["0","1"] for AUX
Mitigation + comparison         ←── per-qubit confusion matrix
        │
        ▼  |P(loss > target)_quantum − P(loss > target)_classical|
metrics.quantum_error.report
```

## 4. Where to start reading

If you are new to the project, read in this order:

1. [`docs/architecture.md`](architecture.md) — system overview, design
   decisions D1–D7, compute budget, risk register.
2. [`docs/CODEBASE.md`](CODEBASE.md) — this document, the file-by-file tour.
3. [`Code/circuitsCRA.py`](code_documentation.md#circuitscrapy) — the
   production `get_expected_probability_circuit` function, the heart of
   the quantum pipeline.
4. [`Code/QSVT.py`](code_documentation.md#qsvtpy) — the QSVT circuit
   class with the QSP→QSVT convention conversion.
5. [`Code/multivariateGCI.py`](code_documentation.md#multivariategcipy) —
   `MultivariateGCI_Linear`, the original Gaussian Conditional
   Independence uncertainty model.
6. [`copula/gaussian.py`](copula_documentation.md) — the drop-in
   replacement for the GCI used by `loader/posterior_factor_copula.py`.
7. [`sbi_pipeline/posterior.py`](sbi_pipeline_documentation.md) — NPE
   / NLE / flow-matching pipelines.
8. [`compiler_backend/heron.py`](compiler_backend_documentation.md) —
   the IBM Heron r3 compiler and pyzx post-optimisation pass.
9. [`experiments/boston_qpu.py`](experiments_documentation.md) — the
   end-to-end driver that compiles, submits, mitigates, and scores the
   circuit on a real IBM QPU.

## 5. Conventions used throughout the codebase

- **Type hints** are used pervasively; float32 is the canonical dtype
  for all risk-arithmetic data, float64 only in numerics that need it.
- **Theta layout** (when shape `(2K + 4,)`) is
  `theta[:K] = factor_loadings`,
  `theta[K:2K] = p_zeros`,
  `theta[2K] = tail_dep`,
  `theta[2K+1] = rho`,
  `theta[2K+2] = nu` (Student-t only),
  `theta[2K+3] = spare`.
  This is the **same** layout for `data.synthetic.SyntheticPortfolioGenerator`,
  all `copula/*`, `loader.posterior_factor_copula`, `qsvt.circuit`, and
  the ground-truth pipelines.
- **LGD** is hard-coded to `0.40` (midpoint of [0.20, 0.60]) in the
  copula simulators; the K=17 Finnish-mortgage dataset overrides this
  with the `Code.dataset_regions.lgd` array (regional loss exposure in
  EUR). See [`Code/dataset_regions.py`](../Code/dataset_regions.py) and
  [`Code/DATASET.md`](../Code/DATASET.md).
- **qiskit-ibm-runtime token resolution** is centralised in
  [`compiler_backend.heron.load_ibm_token`](../compiler_backend/heron.py:64):
  `$IBM_API_KEY` → `$IBM_API_KEY_FILE` → `./.ibm_token` in that order.
  No code should ever read `.ibm_token` directly.
- **Errors that touch the user** (CLI scripts) print to stdout with a
  banner of `=`. Errors that are part of the API (`ValueError`,
  `RuntimeError`) raise normally.

## 6. Re-running everything

The minimum end-to-end run:

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Re-generate the Finnish region dataset (needs internet)
python Code/generate_dataset.py

# 3. Run the demo notebook (no QPU needed)
jupyter execute CRA_QSVT.ipynb

# 4. Replay tests
pytest tests/ -x
```

The full LUMI scaling matrix:

```bash
# From a synced copy on LUMI
bash lumi_deployment/dispatch_all.sh
```

For the live-QPU run:

```bash
export IBM_API_KEY=...
sbatch lumi_deployment/slurm_boston_qpu.sh
```

## 7. License / authorship

The original `Code/` files were authored by **Antonello Aita** and
**Emanuele Dri** (Polito). All other modules are part of the
QSVT4CRA research run, mostly developed by the same team with SBI /
compiler / scaling additions.  See individual file headers for
specific authorship.
