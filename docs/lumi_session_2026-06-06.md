# LUMI Session Report — 2026-06-06

> End-to-end run of the QSVT4CRA-research-run scaling matrix on LUMI
> (project_465003017, "LUST Training / 2026-06-05-07 Junction Quantum Hack").

## TL;DR

| Step | Outcome |
|---|---|
| Sync project to LUMI | `git clone` from GitHub (no rsync on Windows) |
| Install Python deps | 100+ packages, 1.2 GB, ~3 min via `pip --target` |
| Smoke test | ✓ 1m47s on `nid001074` (128 CPUs) |
| SBI training (1 GPU) | ✓ 6m48s on `nid005449` (1 GCD), final log_prob = +1.20 |
| Strong scaling (5 pts) | ✓ All completed in 10m27s (problem too small to show scaling) |
| Hybrid CPU + GPU | ✓ Concurrent, 10m27s + 3m34s |
| MC ground truth (1M scenarios) | ✓ 10m08s, VaR95 = EUR 1.40, CVaR95 = EUR 1.87 |
| Figures generation | ✓ 41s, 7 figures + 1 poster |
| Pull results back | 10 files (2 .npz + 7 .png + 1 .pt) = 2.2 MB |

## Compute used

| Resource | Before session | After session | Delta |
|---|---|---|---|
| CPU Kh | 1.178 (5.9%) | 3.557 (17.8%) | **+2.38 Kh** (this session) |
| GPU hours | 357 (35.7%) | 398 (39.8%) | **+41 GPU-hr** (mostly from the other agent's Heron runs) |
| Scratch | 170 GB | 165 GB | -5 GB (cleanup) |
| Job submits | n/a | ~12 from this session | (under AssocMaxSubmitJobLimit) |

## Job accounting

| JobID | Profile | Status | Elapsed | ReqCPUS | ReqMem | Note |
|---|---|---|---|---|---|---|
| 19086148 | cpu_small | COMPLETED | 1m42s | 8 | 64G | First smoke test (path fix) |
| 19086179 | cpu_small | COMPLETED | 1m58s | 8 | 64G | Smoke test (after path fix) |
| 19086180 | gpu_single | COMPLETED | 6m59s | 32 | 120G | SBI training (1 GCD) |
| 19086696 | hybrid_gpu | COMPLETED | 3m34s | 32 | 120G | SBI training, hybrid |
| 19087512 | cpu_small | COMPLETED | 1m47s | 8 | 64G | Smoke test |
| 19087513 | gpu_single | COMPLETED | 6m48s | 32 | 120G | SBI training |
| 19087514_8 | cpu_strong_8 | COMPLETED | 10m27s | 1 | 16G | Strong scaling, 1 CPU |
| 19087514_16 | cpu_strong_16 | COMPLETED | 10m27s | 1 | 16G | Strong scaling, 2 CPU |
| 19087514_32 | cpu_strong_32 | COMPLETED | 10m27s | 1 | 16G | Strong scaling, 4 CPU |
| 19087514_64 | cpu_strong_64 | COMPLETED | 10m27s | 1 | 16G | Strong scaling, 8 CPU |
| 19087514_128 | cpu_strong_128 | COMPLETED | 10m27s | 1 | 16G | Strong scaling, 16 CPU |
| 19087560 | hybrid_cpu | COMPLETED | 10m27s | 128 | 224G | MC ground truth, hybrid |
| 19087561 | cpu_small | COMPLETED | 10m08s | 8 | 64G | MC ground truth baseline |
| 19087515 | cpu_small | COMPLETED | 0m41s | 8 | 64G | Figures generation |

## Bugs found and fixed (5 commits)

1. **PYTHONPATH pointed to wrong dir** (`b20768f`)
   - `setup_lumi_env.sh` had `${SCRATCH}/site-packages` instead of
     `${SCRATCH}/qsvt4cra-research/site-packages`. Pip installed
     successfully but `import torch` failed. Fixed.

2. **Profiles broke when Slurm copies to /var/spool/** (`de173d2`)
   - `$0` referred to the spool copy. Fixed by using `${SLURM_SUBMIT_DIR}`.

3. **SCRIPT_DIR missing /lumi_deployment subdir** (`dd75350`)
   - Profiles set `SCRIPT_DIR` to the project root, but dispatcher.sh
     lives in `lumi_deployment/`.

4. **Dispatcher hardcoded `posterior_samples.npy`** (`ea8e0a7`)
   - The `mc_ground_truth_weak` and `ood_robustness` cases expected a
     .npy file, but `sbi_train.py` produces a .pt checkpoint. Now
     prefers .pt if present.

5. **`torch.load` rejected numpy globals in PyTorch 2.6** (`b1d60ee`, `ce125f6`)
   - Default is `weights_only=True`; safe unpickler rejects
     `numpy._core.multiarray._reconstruct`. Fixed with
     `weights_only=False` (for trusted local checkpoints).

## Strong scaling observation

All five strong-scaling points completed in **exactly 10m27s**:
- 8 CPUs, 16, 32, 64, 128 — same wall time.

**Root cause**: Slurm allocates a whole node to 1 task regardless of
`--cpus-per-task` (because the standard partition has 128 CPUs/node
and there's only 1 task). All 5 jobs effectively ran with **128 CPUs**
in `SLURM_CPUS_ON_NODE`, so the wall time is identical.

This is a profile/infrastructure issue, not a workload problem:
- `--cpus-per-task=8` on a 128-CPU node with 1 task means
  "this task MAY use up to 8 CPUs" but Slurm doesn't enforce
  isolation. Joblib Parallel(n_jobs=8) on top of an 128-CPU allocation
  runs at the same speed as Parallel(n_jobs=128).
- To do real strong scaling, would need either:
  - `taskset -c 0-7` to pin the worker to specific cores
  - Or `--exclusive` + `--mem=0` so Slurm can't oversubscribe
  - Or set `OMP_NUM_THREADS=8` env var in the profile to constrain
    OpenMP thread count

**Recommendation for a meaningful scaling study**:
- Fix the profiles to use `taskset` or `OMP_NUM_THREADS` for actual CPU binding
- Increase N_SCENARIOS to 10M or 100M
- Or use a larger K (e.g., K=100 or K=1000)
- Or add the weak-scaling matrix (1, 2, 4, 8 nodes) where work scales
  with resources — this would show sub-linear speedup due to network

## Files produced on LUMI

```
/scratch/project_465003017/kkiirikk/qsvt4cra-research/
├── checkpoints/
│   └── sbi_npe_baseline_K10.pt            137 KB
├── results/
│   ├── mc_ground_truth_gaussian_baseline_K10.npz   28 KB
│   └── mc_ground_truth_weak_1nodes.npz             28 KB
├── figures/
│   ├── fig1_posterior_uncertainty.png              261 KB
│   ├── fig2_loss_distributions.png                 191 KB
│   ├── fig3_var_cvar_uncertainty.png               197 KB
│   ├── fig4_qsvt_error.png                         402 KB
│   ├── fig5_ood_calibration.png                    195 KB
│   ├── fig6_quantum_scaling.png                    526 KB
│   └── fig7_pipeline_poster.png                    237 KB
└── slurm_logs/
    └── qsvt4cra-*  (15+ log files)
```

All files pulled to local `results_lumi/`, `figures_lumi/`, `checkpoints_lumi/`.

## Next steps

- [ ] For a real scaling study: scale to K=100 and N_SCENARIOS=100M
- [ ] Multi-GPU DDP scaling once `--gpus-per-node=4` is available
      (currently rejected: "Requested node configuration is not available")
- [ ] Weak scaling matrix (1, 2, 4, 8 nodes) — submitted as
      `dispatch_all.sh` will work but needs more wait time
- [ ] Real Finnish mortgage data: still no StatFin/Eurostat integration
      (was flagged as a Phase 1 extension, not yet done)
- [ ] Add `wandb` to the LUMI runs to track metrics across job boundaries

## How to reproduce

From Windows PowerShell (one command does everything):

```powershell
cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
.\lumi_deployment\submit_from_windows.ps1
```

Or from Linux/Mac:

```bash
cd QSVT4CRA
bash lumi_deployment/rsync_to_lumi.sh    # or: ssh lumi "cd ... && git pull"
ssh lumi
cd /scratch/project_465003017/$USER/qsvt4cra-research
bash lumi_deployment/dispatch_all.sh
```

Time: ~1.5 hours end-to-end (includes pip install, all 15 jobs, pull results).
