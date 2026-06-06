# LUMI Submission Guide for QSVT4CRA Research Run

This guide walks through the full submission pipeline for the
Posterior-Propagated Factor-Copula QSVT research run on LUMI.

> **TL;DR (Windows host)**:
> ```powershell
> cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
> .\lumi_deployment\submit_from_windows.ps1 -SmokeOnly    # 5-min smoke
> .\lumi_deployment\submit_from_windows.ps1                # full matrix
> ```
> **TL;DR (Linux/Mac host)**:
> ```bash
> cd QSVT4CRA
> bash lumi_deployment/rsync_to_lumi.sh
> ssh lumi
> cd /scratch/project_465003017/$USER/qsvt4cra-research
> bash lumi_deployment/setup_lumi_env.sh
> pip install --no-cache-dir --target=./site-packages -r requirements.txt
> bash lumi_deployment/dispatch_all.sh
> ```

---

## 1. Project context

| Item | Value |
|---|---|
| LUMI user | `kkiirikk` |
| Project | `project_465003017` (LUST Training / Junction Quantum Hack 2026-06) |
| Time left | 96 days (allocation ends 2026-09-10) |
| CPU Kh used | 1.178 of 20 Kh (5.9%) — wide open |
| GPU Kh budget | 50–150 hr (project brief) |
| Scratch quota | 170 GB used of 55 TB |
| Home quota | **OVER** (101K of 100K files) — keep work in scratch |
| Job submit limit | Hit `AssocMaxSubmitJobLimit` on ad-hoc srun — use job arrays |

---

## 2. Architecture overview

```
Local (Windows)                       LUMI (scratch)
═══════════════                       ══════════════
QSVT4CRA/                            /scratch/project_465003017/
├── lumi_deployment/                 └── kkiirikk/
│   ├── setup_lumi_env.sh                └── qsvt4cra-research/
│   ├── rsync_to_lumi.sh                     ├── Code/             (untouched paper repo)
│   ├── dispatcher.sh    ◄──────────┐        ├── copula/
│   ├── profiles/        ◄──────────┤        ├── data/
│   │   ├── cpu_small.sh             │        ├── experiments/
│   │   ├── cpu_med.sh               │        ├── qsvt/
│   │   ├── cpu_weak_4.sh            │        ├── sbi_pipeline/
│   │   ├── gpu_single.sh            │        ├── simulator/
│   │   └── ...                      │        ├── scale_runner.py
│   ├── dispatch_all.sh    ◄─────────┘        ├── site-packages/   (pip --target)
│   └── submit_from_windows.ps1               ├── requirements.txt
├── copula/, data/, ...                        ├── checkpoints/
├── scale_runner.py                            ├── results/
├── requirements.txt                           ├── figures/
└── Code/  (untouched)                         └── slurm_logs/
```

The `dispatcher.sh` is the single source of truth for the experiment case
statement. Each profile in `profiles/` is a self-contained Slurm script
that sources `dispatcher.sh` and sets `EXPERIMENT` / `K` / `REGIME` / etc.

---

## 3. Step-by-step submission

### 3.1 One-time Windows setup

1. Verify OpenSSH client:
   ```powershell
   ssh -V
   ```
   (Windows 10 1809+ ships OpenSSH; no install needed.)

2. Set up SSH config alias for LUMI:
   ```powershell
   notepad $HOME\.ssh\config
   ```
   Add:
   ```
   Host lumi
     HostName lumi.csc.fi
     User kkiirikk
     ServerAliveInterval 60
     ServerAliveCountMax 3
   ```

3. Add your CSC/MyAccessID SSH key (if not already on LUMI):
   - MyAccessID → https://mms.MyAccessID.org/ → "Public SSH key"
   - Or copy from local:
     ```powershell
     type $HOME\.ssh\id_ed25519.pub
     ```

### 3.2 One-time LUMI environment setup

```bash
ssh lumi
cd /scratch/project_465003017/$USER

# Bootstrap (idempotent)
mkdir -p qsvt4cra-research/{Code,site-packages,results,checkpoints,figures,slurm_logs}
```

### 3.3 Sync project (every time you change code)

**From Windows (PowerShell)**:
```powershell
cd C:\Users\Käyttäjä\Documents\projects\QSVT4CRA
.\lumi_deployment\submit_from_windows.ps1 -SkipInstall -SmokeOnly
```

**From Windows (one command does everything)**:
```powershell
.\lumi_deployment\submit_from_windows.ps1 -SmokeOnly
```

**From Linux/Mac**:
```bash
cd QSVT4CRA
bash lumi_deployment/rsync_to_lumi.sh
```

### 3.4 Install Python deps (one-time, then on demand)

```bash
ssh lumi
cd /scratch/project_465003017/$USER/qsvt4cra-research
bash lumi_deployment/setup_lumi_env.sh
pip install --no-cache-dir --target=./site-packages -r requirements.txt
```

Expected installs (~5–10 min):
- `torch` (CPU + ROCm 6.3.4 wheels)
- `sbi==0.22.0`
- `qiskit==1.1.2`
- `pyqsp==0.2.0`
- `numpyro`, `joblib`, `wandb`, `matplotlib`, `scipy`

### 3.5 Submit the scaling matrix

**Smoke test first** (5 min, 1 node, 8 CPU):
```bash
ssh lumi
cd /scratch/project_465003017/$USER/qsvt4cra-research
EXPERIMENT=smoke_test sbatch lumi_deployment/profiles/cpu_small.sh
# Wait for completion
squeue -u $USER
# Check output
ls -lt slurm_logs/ | head
cat slurm_logs/qsvt4cra-cpu-small-*.out
```

**Full matrix** (after smoke passes):
```bash
bash lumi_deployment/dispatch_all.sh
```

This submits ~15 jobs in dependency order:
1. **Phase 0** — smoke test (1 node, 8 CPU, 5 min)
2. **Phase 1** — SBI training (1 GPU, 30 min)
3. **Phase 2** — CPU strong scaling (1 node, 5 runs: 8, 16, 32, 64, 128 CPUs, 2h each)
4. **Phase 3** — CPU weak scaling (1, 2, 4, 8 nodes, 4h each)
5. **Phase 4** — GPU scaling (1, 2, 4 GPUs, 4h each)
6. **Phase 5** — Hybrid CPU+GPU (concurrent, 4h)
7. **Phase 6** — Figure generation (waits for all, 5 min)

**Total compute budget** (worst case):
- GPU: ~50 GPU-hr (3 GPU jobs × 4h × ~4 GPUs each)
- CPU: ~250 CPU-hr (4 weak × 4h × 256 + 5 strong × 2h × 128) — well within 500 CPU-hr budget
- Wall time: 4–6 hours (most jobs run in parallel after SBI)

### 3.6 Monitor

```bash
# Live status
ssh lumi "squeue -u $USER"

# Historical / accounting
ssh lumi "sacct -X -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,ReqGPU,ReqCPUS --starttime 2026-06-06"

# Per-job log
ssh lumi "cat /scratch/project_465003017/$USER/qsvt4cra-research/slurm_logs/qsvt4cra-cpu-med-*.out"

# All logs, sorted by time
ssh lumi "ls -lt /scratch/project_465003017/$USER/qsvt4cra-research/slurm_logs/ | head -20"
```

### 3.7 Pull results back

```bash
# From Windows PowerShell
scp kkiirikk@lumi.csc.fi:/scratch/project_465003017/kkiirikk/qsvt4cra-research/results/*.npz ./results/
scp kkiirikk@lumi.csc.fi:/scratch/project_465003017/kkiirikk/qsvt4cra-research/figures/*.png ./figures/
```

Or use the existing `rsync_to_lumi.sh` in reverse (it works for either direction).

---

## 4. Scaling matrix at a glance

| Profile | Nodes | CPUs/task | GPUs | Time | Purpose |
|---|---|---|---|---|---|
| `cpu_small` | 1 | 8 | 0 | 30m | Smoke test |
| `cpu_med` | 1 | 256 | 0 | 4h | Single-node MC |
| `cpu_large` | 1 | 256 | 0 | 8h | Long MC (1e7) |
| `cpu_strong_8/16/32/64/128` | 1 | 8–128 | 0 | 2h each | Strong scaling (Amdahl) |
| `cpu_weak_2/4/8` | 2/4/8 | 256 | 0 | 4h each | Weak scaling |
| `gpu_single` | 1 | 32 | 1 | 4h | NPE training |
| `gpu_strong_4` | 1 | 32 | 4 | 4h | Multi-GPU DDP |
| `gpu_multi_2/4` | 2/4 | 32 | 1 each | 4h | Multi-node DDP |
| `hybrid_cpu` | 1 | 128 | 0 | 4h | MC half |
| `hybrid_gpu` | 1 | 32 | 1 | 4h | SBI half |

---

## 5. Compute budget validation

| Resource | Budget | Plan | Margin |
|---|---|---|---|
| CPU Kh | 500 (5000 hr × 100 users, total 20 Kh project) | ~3 Kh | 6.7× |
| GPU hr | 50–150 | ~50 hr | 1–3× |
| Job submits | ~20/user | ~15 (incl. arrays) | ✓ |
| Scratch GB | 55 TB | <5 GB | ✓ |
| Wall time | 96 days | 1 day | ✓ |

---

## 6. Troubleshooting

### "AssocMaxSubmitJobLimit"
You submitted too many jobs in a window. The dispatcher uses **job arrays**
to pack the strong-scaling and weak-scaling sweeps into a few submissions.
If you hit the limit anyway, wait an hour or use `scancel` to free slots.

### "ModuleNotFoundError: No module named 'sbi'"
`pip install --no-cache-dir --target=./site-packages -r requirements.txt`
on LUMI (after `source lumi_deployment/setup_lumi_env.sh`).

### "Address already in use" (DDP)
Two jobs trying to use the same `MASTER_PORT`. Use `--export=ALL` and the
dispatcher sets a unique port via `MASTER_PORT=$((29500 + SLURM_JOB_ID % 100))`.

### "Out of memory" on GPU node
Reduce `--gpus-per-node` to 1 or `--n-simulations` to 100 (in `dispatcher.sh`).

### "Cannot find Code/..."
`rsync_to_lumi.sh` only syncs top-level project dirs and `Code/`. The
script must be run from the project root, or `LOCAL_ROOT` must be set.

### Job stuck in `PENDING` forever
Check `squeue -j <jobid> -o "%.18i %.10P %.20j %.8T %.10M %.10l %R"` —
the `REASON` column tells you why. Common:
- `ReqNodeNotAvail`: requested node type busy; wait or reduce nodes
- `Priority`: queue wait; check `sprio -j <jobid>` for priority factors

### Cross-FS rename error on `pip install`
Lustre vs NVMe. Workaround: `pip install --no-cache-dir --target=./site-packages`
puts everything in scratch; see `docs/hardware_notes.md`.

### Home quota over
Already over (101K/100K). DO NOT `pip install` to home; always use
`--target=./site-packages` in scratch. The `setup_lumi_env.sh` script
sets `PYTHONUSERBASE` to scratch to prevent this.

---

## 7. Files reference

| File | Purpose |
|---|---|
| `lumi_deployment/setup_lumi_env.sh` | Module + scratch + PYTHONPATH bootstrap |
| `lumi_deployment/rsync_to_lumi.sh` | Local → LUMI scratch sync |
| `lumi_deployment/dispatcher.sh` | Experiment case statement (shared) |
| `lumi_deployment/slurm_qsvt4cra_research.sh` | Dispatcher: picks profile + submits |
| `lumi_deployment/dispatch_all.sh` | Master orchestrator (submits all phases) |
| `lumi_deployment/profiles/*.sh` | Slurm profile presets (15 total) |
| `lumi_deployment/submit_from_windows.ps1` | One-command submit from Windows |
| `scale_runner.py` | Multi-node launcher (joblib + torch.distributed) |
| `experiments/sbi_train.py` | SBI training CLI (NPE/NLE/cMAF) |
| `docs/lumi_submission_guide.md` | This document |
| `docs/lumi_scaling_strategy.md` | What/why of the scaling matrix |
| `docs/hardware_notes.md` | LUMI module stack + gotchas |
