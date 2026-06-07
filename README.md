# QSVT4CRA

Here, we implement a real estate risk analysis with quantum singular value transformation (QSVT, https://arxiv.org/abs/1806.01838) based on the paper **"Implementing Credit Risk Analysis with Quantum Singular Value Transformation"** (https://arxiv.org/abs/2507.19206). As a test case, we use data from the finnish real estate market, divided into 17 geographical regions. Our code shows promising results in noiseless QPU simulators on LUMI, but fails to show good results on actual quantum computers due to the noise they exhibit.

The `Code/` package is the original implementation by
**Antonello Aita** (`antonello.aita@gmail.com`) and
**Emanuele Dri** (`emanuele.dri@polito.it`), Politecnico di
Torino. All other modules (`copula/`, `data/`, `loader/`,
`qsvt/`, `metrics/`, `simulator/`, `sbi_pipeline/`,
`compiler_backend/`, `experiments/`, `lumi_deployment/`,
`scale_runner.py`) form the QSVT4CRA research run that
layers on top.

See [`docs/CODEBASE.md`](docs/CODEBASE.md) for a full
code-level tour and [`docs/architecture.md`](docs/architecture.md)
for the system design (AI generated).

What follows is an AI generated summary of the main functionalities included in the package.

## Installation

```bash
pip install -r requirements.txt
```

Optionally refresh the K=17 Finnish region dataset (needs
internet to fetch from `pxdata.stat.fi`):

```bash
python Code/generate_dataset.py
```

Run the tests to verify the install:

```bash
pytest tests/
```

## Quickstart (no QPU)

```bash
jupyter execute CRA_QSVT.ipynb
```

The notebook demonstrates the minimal principles the QSVT algorithm is based on.

## Pipelines

All pipelines are CLI entry points under `experiments/`. Run
them with `python -m experiments.<name>`.

| Pipeline | Module | Purpose |
|---|---|---|
| **Phase 1 — SBI training** | `experiments.sbi_train` | Train an NPE / NLE / flow-matching posterior over factor-copula parameters |
| **Phase 3 — MC ground truth** | `experiments.mc_ground_truth` | Massive-MC VaR/CVaR benchmark |
| **Phase 4 — QSVT degree sweep** | `experiments.qsvt_sweep` | QSVT approximation error vs polynomial degree |
| **Phase 4 — Heron simulation** | `experiments.heron_simulation` | Aer simulation with IBM Heron r3 calibration data (no QPU access) |
| **Phase 4 — Real QPU** | `experiments.boston_qpu` | Compile, submit, and benchmark on the real IBM QPU |
| **Phase 4 — Sweep analysis** | `experiments.analyze_degree_sweep` | Post-process degree-sweep job results |
| **Phase 4 — Job status** | `experiments.check_qpu_jobs` | Poll or fetch recent IBM Quantum jobs |
| **Phase 5 — OOD robustness** | `experiments.ood_robustness` | Distribution-shift experiment (point-estimate GCI vs posterior factor copula) |
| **Phase 6 — Resource scaling** | `experiments.resource_scaling` | Qubits / depth / T-count vs portfolio size K |

### Phase 1 — SBI training

```bash
python -m experiments.sbi_train \
    --method npe \
    --n-simulations 1000 \
    --n-rounds 10 \
    --device cuda \
    --K 10 \
    --regime baseline \
    --output checkpoints/sbi_npe_baseline.pt
```

Methods: `npe` (default), `nle`, `fm` (flow-matching / cMAF).

### Phase 3 — MC ground truth

```bash
python -m experiments.mc_ground_truth \
    --posterior-checkpoint checkpoints/sbi_npe_baseline_K10.pt \
    --n-scenarios 1000000 \
    --copula gaussian \
    --regime baseline \
    --K 10 \
    --output results/ground_truth_gaussian_baseline.npz
```

Use `--streaming` for `n_scenarios ≥ 1e7`.

### Phase 4 — QSVT degree sweep (Aer, no QPU)

```bash
python -m experiments.qsvt_sweep \
    --degrees 16 32 64 128 256 512 1024 \
    --posterior-checkpoint posterior_samples.npz \
    --K 10 \
    --target-loss 0.5 \
    --n-shots 10000 \
    --output results/qsvt_sweep.npz
```

### Phase 4 — IBM Heron calibration-aware simulation

Requires an IBM Quantum token. Set it first:

```bash
export IBM_API_KEY=...
```

```bash
python -m experiments.heron_simulation \
    --backend-name ibm_boston \
    --degree 4 \
    --shots 1024 \
    --target-loss-fraction 0.5 \
    --output results/heron_k17_simulation.json
```

### Phase 4 — Real IBM QPU

```bash
export IBM_API_KEY=...
python -m experiments.boston_qpu \
    --backend-name ibm_boston \
    --dataset-source finnish \
    --k 17 \
    --degree 8 \
    --shots 2048 \
    --target-loss-fraction 0.5 \
    --n-classical-scenarios 200000 \
    --output results/boston_qpu_k17.json
```

Reads the AUX qubit (the QSVT projector answer bit). Uses
real Chebyshev-QSP phases from `pyqsp`. Optional
`--use-pyzx` for an extra `pyzx.basic_optimization` pass;
default `--no-mitigation` skips the per-qubit readout
calibration.

### Phase 5 — Out-of-distribution robustness

```bash
python -m experiments.ood_robustness \
    --posterior-checkpoint checkpoints/sbi_npe_baseline_K10.pt \
    --test-regimes baseline housing_crash rate_shock_0.5 rate_shock_1.5 unemployment liquidity \
    --n-posterior-samples 1000 \
    --n-scenarios 100000 \
    --K 10 \
    --output results/ood_robustness_K10.npz \
    --plot figures/ood_robustness_K10.png
```

### Phase 6 — Resource scaling

```bash
python -m experiments.resource_scaling \
    --n-loans 10 50 100 500 1000 \
    --degree 64 \
    --output figures/resource_scaling.png \
    --csv results/resource_scaling.csv
```

## LUMI HPC deployment

For the LUMI-G MI250X supercomputer, see
[`lumi_deployment/README.md`](lumi_deployment/README.md).
The full scaling matrix is dispatched with:

```bash
bash lumi_deployment/dispatch_all.sh
```

Individual LUMI jobs (e.g., the K=17 QPU benchmark) are
submitted with:

```bash
sbatch lumi_deployment/slurm_boston_qpu.sh
```

## Project layout

```
Code/                    # Original QSVT-on-GCI prototype
copula/                  # Factor-copula simulators (Gaussian / Student-t / D-vine / low-rank)
data/                    # Synthetic portfolio, stress regimes, real-data loader (stub)
loader/                  # Quantum state-preparation loaders
qsvt/                    # QSVT circuits and phase-sequence approximators
metrics/                 # VaR / CVaR / MC ground truth / quantum-vs-classical error
simulator/               # Forward simulators (NumPy + JAX)
sbi_pipeline/            # NPE / NLE / flow-matching SBI pipelines
compiler_backend/        # IBM Heron r3 calibration-aware compilation
experiments/             # CLI entry points (one per phase)
tests/                   # pytest suite
lumi_deployment/         # Slurm profiles, dispatchers, env setup
docs/                    # Architecture + per-module documentation
scale_runner.py          # Multi-node launcher abstraction
CRA_QSVT.ipynb           # Primary demo notebook
```

## License

The `Code/` package is the original work of Aita & Dri
(Politecnico di Torino). All other modules are released as
part of the QSVT4CRA research run; see individual file
headers for authorship.
