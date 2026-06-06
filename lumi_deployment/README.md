# LUMI Deployment for QSVT4CRA-Research-Run

This directory contains the LUMI HPC deployment artifacts for the
**Posterior-Propagated Factor-Copula QSVT for Apartment Loan Portfolio Risk**
research run.

> Ported from `projects/JunctionHackathon` and `projects/JunctionHackathon-lumi`,
> which share the same LUMI allocation (`project_465003017`).

## Files

| File | Purpose |
|---|---|
| `setup_lumi_env.sh` | Module stack + scratch bootstrap |
| `rsync_to_lumi.sh` | Local → LUMI sync helper |
| `dispatcher.sh` | Shared experiment case statement (sourced by profiles) |
| `slurm_qsvt4cra_research.sh` | Dispatcher entrypoint (picks profile + submits) |
| `dispatch_all.sh` | Master orchestrator: submits the full scaling matrix |
| `profiles/*.sh` | Slurm profile presets (CPU/GPU/Hybrid, 17 total) |
| `submit_from_windows.ps1` | One-command submit from Windows PowerShell |

## Quick start

### From Windows PowerShell

```powershell
cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
.\lumi_deployment\submit_from_windows.ps1 -SmokeOnly    # 5-min smoke
.\lumi_deployment\submit_from_windows.ps1                # full scaling matrix
```

### From Linux/Mac

```bash
cd QSVT4CRA

# 1. Sync to LUMI
bash lumi_deployment/rsync_to_lumi.sh

# 2. Install deps (one-time)
ssh kkiirikk@lumi.csc.fi
cd /scratch/project_465003017/$USER/qsvt4cra-research
bash lumi_deployment/setup_lumi_env.sh
pip install --no-cache-dir --target=./site-packages -r requirements.txt

# 3. Run smoke test
EXPERIMENT=smoke_test sbatch lumi_deployment/profiles/cpu_small.sh

# 4. Submit full matrix
bash lumi_deployment/dispatch_all.sh
```

## Profile catalog

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

## Compute budget (per project brief)

| Resource | Available | Plan | Margin |
|---|---|---|---|
| GPU-hours | 2000 | 50–150 | ✓ |
| CPU-hours | Large (20 Kh project) | <500 | ✓ |
| Wall days | 96 | <1 | ✓ |
| Job submits | ~20/user (rolling limit) | ~15 | ✓ |

## See also

- `docs/lumi_submission_guide.md` — step-by-step submission
- `docs/lumi_scaling_strategy.md` — what/why of the scaling matrix
- `docs/hardware_notes.md` — LUMI module stack and known gotchas
- `scale_runner.py` — multi-node launcher abstraction
- `experiments/sbi_train.py` — SBI training CLI entry point
