# `lumi_deployment/` — LUMI HPC deployment

> Slurm infrastructure for the LUMI-G MI250X supercomputer.
> Profiles, dispatchers, environment setup, and a
> Windows-PowerShell submission helper.

## Files

| File | Purpose |
|---|---|
| [`README.md`](#readmemd) | User-facing quickstart |
| [`setup_lumi_env.sh`](#setup_lumi_envsh) | Module stack + scratch bootstrap |
| [`rsync_to_lumi.sh`](#rsync_to_lumish) | Local → LUMI sync helper |
| [`dispatcher.sh`](#dispatchersh) | Shared experiment case statement |
| [`slurm_qsvt4cra_research.sh`](#slurm_qsvt4cra_researchsh) | Dispatcher entrypoint |
| [`slurm_heron_simulation.sh`](#slurm_heron_simulationsh) | LUMI wrapper for `experiments/heron_simulation.py` |
| [`slurm_boston_qpu.sh`](#slurm_boston_qpush) | LUMI wrapper for `experiments/boston_qpu.py` |
| [`dispatch_all.sh`](#dispatch_allsh) | Master orchestrator for the full scaling matrix |
| [`submit_from_windows.ps1`](#submit_from_windowspowershellsh) | One-command submit from Windows PowerShell |
| [`profiles/*.sh`](#profilessh) | 17 Slurm profile presets |

The catalog of profiles:

| Profile | Nodes | CPUs/task | GPUs | Time | Use case |
|---|---|---|---|---|---|
| `cpu_small` | 1 | 8 | 0 | 30m | Smoke test |
| `cpu_med` | 1 | 256 | 0 | 4h | Single-node MC |
| `cpu_large` | 1 | 256 | 0 | 8h | Long MC (1e7 scenarios) |
| `cpu_strong_8` | 1 | 8 | 0 | 2h | Strong scaling baseline |
| `cpu_strong_16` | 1 | 16 | 0 | 2h | Strong scaling |
| `cpu_strong_32` | 1 | 32 | 0 | 2h | Strong scaling |
| `cpu_strong_64` | 1 | 64 | 0 | 2h | Strong scaling |
| `cpu_strong_128` | 1 | 128 | 0 | 2h | Strong scaling |
| `cpu_weak_2` | 2 | 256 | 0 | 4h | Weak scaling (2×) |
| `cpu_weak_4` | 4 | 256 | 0 | 4h | Weak scaling (4×) |
| `cpu_weak_8` | 8 | 256 | 0 | 4h | Weak scaling (8×) |
| `gpu_single` | 1 | 32 | 1 | 4h | SBI training (1 GCD) |
| `gpu_strong_4` | 1 | 32 | 4 | 4h | Multi-GPU DDP (1 node) |
| `gpu_multi_2` | 2 | 32 | 2 | 4h | Multi-node DDP (2 nodes) |
| `gpu_multi_4` | 4 | 32 | 4 | 4h | Multi-node DDP (4 nodes) |
| `hybrid_cpu` | 1 | 128 | 0 | 4h | MC half (concurrent with GPU) |
| `hybrid_gpu` | 1 | 32 | 1 | 4h | SBI half (concurrent with CPU) |

Compute budget (per project brief):

| Resource | Available | Plan | Margin |
|---|---|---|---|
| GPU-hours | 2000 | 50–150 | ✓ |
| CPU-hours | Large (20 Kh project) | <500 | ✓ |
| Wall days | 96 | <1 | ✓ |
| Job submits | ~20/user (rolling limit) | ~15 | ✓ |

---

## `README.md`

User-facing quickstart. Two paths:

### From Windows PowerShell

```powershell
cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
.\lumi_deployment\submit_from_windows.ps1 -SmokeOnly    # 5-min smoke
.\lumi_deployment\submit_from_windows.ps1                # full scaling matrix
```

### From Linux/Mac

```bash
cd QSVT4CRA
bash lumi_deployment/rsync_to_lumi.sh
ssh kkiirikk@lumi.csc.fi
cd /scratch/project_465003017/$USER/qsvt4cra-research
bash lumi_deployment/setup_lumi_env.sh
pip install --no-cache-dir --target=./site-packages -r requirements.txt
EXPERIMENT=smoke_test sbatch lumi_deployment/profiles/cpu_small.sh
bash lumi_deployment/dispatch_all.sh
```

---

## `setup_lumi_env.sh`

Bootstrap a LUMI compute environment. Sourced by all
profile scripts and the dispatcher.

Steps:

1. **Module stack** (verified 2026-06):
   `LUMI/25.09`, `partition/G`, `rocm` (6.4.4),
   `cray-python/3.11.7`.
2. **Scratch directory**:
   `SCRATCH=/scratch/${SLURM_JOB_ACCOUNT:-project_465003017}/${USER}`.
3. **PYTHONPATH**: prepends
   `$SCRATCH/qsvt4cra-research/site-packages` so user-installed
   pip packages are importable.
4. **JAX / ROCm**: sets `JAX_PLATFORMS=rocm` if
   `site-packages/jax` exists, else unsets `JAX_PLATFORMS`
   (fall back to default).
5. **Cray MPICH**: sets `USE_SRUN=1` (LUMI has no `mpirun`).
6. **Slurm vars**: prints job ID, node list, CPUs/node,
   GPUs/node if running inside a job.
7. **Sanity printout** of the key env vars.

---

## `rsync_to_lumi.sh`

`rsync` the local repo to LUMI's scratch. Excludes
`.venv`, `__pycache__`, `.pytest_cache`, `.git`, results,
figures, and the IBM token file. The destination is
`/scratch/project_465003017/$USER/qsvt4cra-research/`.

Run from the local repo root: `bash lumi_deployment/rsync_to_lumi.sh`.

---

## `dispatcher.sh`

The shared experiment case statement. Sourced by every
profile (`profiles/*.sh`). Picks the experiment to run based
on the `EXPERIMENT` env var and dispatches to the right CLI.

Supported `EXPERIMENT` values:

| Value | What it runs |
|---|---|
| `sbi_train` | `experiments.sbi_train` |
| `mc_ground_truth` | `experiments.mc_ground_truth` |
| `mc_ground_truth_weak` | `experiments.mc_ground_truth` with `N_SCENARIOS = 1e6 · N_NODES` |
| `qsvt_sweep` | `experiments.qsvt_sweep` |
| `ood_robustness` | `experiments.ood_robustness` |
| `resource_scaling` | `experiments.resource_scaling` |
| `figures` | `experiments.make_figures` |
| `smoke_test` | 30-second environment check |

Each branch sources the relevant env vars (`K`, `REGIME`,
`COPULA`, `N_SCENARIOS`, `N_SIMULATIONS`, `N_ROUNDS`,
`SBI_METHOD`, `SBI_DEVICE`, `SEED`, etc.) before invoking
the Python CLI.

The `smoke_test` branch is the recommended first submission:
it prints `python --version`, `torch.__version__`,
`numpy.__version__`, `sbi.__version__`, `qiskit.__version__`,
`joblib.__version__`, and the detected `scale_runner`
environment, then exits successfully. If any of those
imports fail, the test surfaces the error early.

---

## `slurm_qsvt4cra_research.sh`

The dispatcher entrypoint. Sets `PROFILE_NAME`,
sources `dispatcher.sh`, and runs the experiment selected
by `EXPERIMENT`.

The `#SBATCH` directives are minimal (single node, single
task); the actual resource allocation comes from the
profile that submits this script. Override `EXPERIMENT`,
`K`, `REGIME`, `COPULA`, `SEED`, `N_SCENARIOS`,
`N_SIMULATIONS`, `N_ROUNDS`, `SBI_METHOD`, `SBI_DEVICE`
via Slurm `--export`.

---

## `slurm_heron_simulation.sh`

LUMI wrapper for `experiments.heron_simulation.py`. Requests
1 node × 56 CPUs × 1 GPU × 480 GB RAM × 4h. Validates that
`$IBM_API_KEY` is set, then runs the Python CLI.

---

## `slurm_boston_qpu.sh`

LUMI wrapper for `experiments.boston_qpu.py`. Requests
1 node × 16 CPUs × 0 GPUs × 64 GB RAM × 4h. Validates the
IBM token via `compiler_backend.heron.load_ibm_token`'s
fallback chain, then runs the Python CLI.

The lower CPU and memory budget (vs `slurm_heron_simulation.sh`)
is because the Boston QPU job is dominated by the queue time,
not the local compute; the actual execution happens on the
remote IBM QPU.

---

## `dispatch_all.sh`

Master orchestrator. Iterates over the full scaling matrix
and submits one job per profile + experiment combination.

The matrix is configured to satisfy the project's
`~15 jobs/user` rolling limit on LUMI and the `<500 CPU-hr`,
`50–150 GPU-hr` budget.

---

## `submit_from_windows.ps1`

PowerShell wrapper for the Windows-side workflow:

```powershell
cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
.\lumi_deployment\submit_from_windows.ps1 -SmokeOnly    # 5-min smoke
.\lumi_deployment\submit_from_windows.ps1                # full matrix
```

Handles `rsync`, the SSH password prompt, and the
`EXPERIMENT=... sbatch ...` invocations in one go.

---

## `profiles/*.sh`

The 17 Slurm profile presets. Each file:

1. Sets `PROFILE_NAME` and the relevant env vars.
2. Has its own `#SBATCH` directives.
3. Sources `lumi_deployment/dispatcher.sh`.

For example, `cpu_small.sh` (5-min smoke):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=qsvt4cra-smoke
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

PROFILE_NAME="cpu_small"
EXPERIMENT="smoke_test"
export EXPERIMENT PROFILE_NAME
bash "${SLURM_SUBMIT_DIR}/lumi_deployment/dispatcher.sh"
```

Each profile is a single bash script that any of the
scaling experiments can be dispatched through. The matrix
in `dispatch_all.sh` picks `PROFILE_NAME × EXPERIMENT`
combinations.

---

## See also

- [`docs/lumi_submission_guide.md`](lumi_submission_guide.md)
  — step-by-step submission (pre-existing).
- [`docs/lumi_scaling_strategy.md`](lumi_scaling_strategy.md)
  — what/why of the scaling matrix (pre-existing).
- [`docs/hardware_notes.md`](hardware_notes.md) — LUMI module
  stack and known gotchas (pre-existing).
- [`scale_runner.py`](../scale_runner.py) — multi-node
  launcher abstraction (see
  [`docs/scale_runner_documentation.md`](scale_runner_documentation.md)).
- [`experiments/sbi_train.py`](experiments_documentation.md#sbi_trainpy)
  — SBI training CLI entry point.
