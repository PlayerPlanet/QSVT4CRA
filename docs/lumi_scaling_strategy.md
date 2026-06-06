# LUMI Scaling Strategy for QSVT4CRA Research Run

> **Why this exists**: The pipeline has 3 distinct workload classes
> (SBI training, MC ground truth, figure rendering) with very different
> parallelization patterns. A single Slurm profile wastes cycles. This
> document explains the matrix and the science behind each row.

---

## 1. Workload characterization

| Workload | Scales on | Bottleneck | Parallelization | Ideal scale |
|---|---|---|---|---|
| **SBI training** (NPE/NLE/cMAF) | GPU (ROCm) | Matrix ops in `posterior_nn` | `torch.distributed` (DDP) | 1–4 GPUs, batch 100–1000 |
| **MC ground truth** | CPU (NumPy) | Vectorized inner loops | `joblib` per posterior sample | 64–256 CPUs (1 node) |
| **QSVT degree sweep** | CPU (small sims) | Sequential poly construction | `joblib` over degrees | 8–16 CPUs |
| **OOD robustness** | CPU | Per-regime + per-posterior-sample | `joblib` nested | 32–64 CPUs |
| **Resource scaling** | CPU | Closed-form math | Sequential is fine | 1 CPU |
| **Figure generation** | CPU | Matplotlib rendering | Sequential is fine | 1 CPU |

---

## 2. Strong vs weak scaling — when to use which

### 2.1 Strong scaling (fixed problem, varying resources)
Use when:
- You have a fixed scientific question ("how fast can I compute VaR for
  this 1e6-scenario portfolio?")
- You want to compute **speedup = t(1) / t(N)** and **efficiency = speedup / N**
- The work has small fixed overheads and scales sublinearly

**QSVT4CRA strong-scaling study**:
| Profile | CPUs | Expected time | Speedup vs 8-CPU | Efficiency |
|---|---|---|---|---|
| `cpu_strong_8` | 8 | ~600 s (baseline) | 1.00× | 100% |
| `cpu_strong_16` | 16 | ~320 s | 1.88× | 94% |
| `cpu_strong_32` | 32 | ~180 s | 3.33× | 83% |
| `cpu_strong_64` | 64 | ~110 s | 5.45× | 68% |
| `cpu_strong_128` | 128 | ~80 s | 7.50× | 47% |

Amdahl's law: at 100 CPUs, ~50% efficiency from sequential parts (copula init,
per-sample VaR/CVaR kernel). Above 256 CPUs, diminishing returns.

### 2.2 Weak scaling (work grows with resources)
Use when:
- The real question is "what's the largest portfolio I can analyze in
  a given wall time?"
- You want **constant time per unit work** as you scale
- Monte Carlo is the canonical example: 2x cores → 2x scenarios → same time

**QSVT4CRA weak-scaling study**:
| Profile | Nodes | CPUs | Scenarios | Expected time |
|---|---|---|---|---|
| `cpu_weak_2` (1 node) | 1 | 256 | 1e6 | 600 s |
| `cpu_weak_2` | 2 | 512 | 2e6 | 620 s |
| `cpu_weak_4` | 4 | 1024 | 4e6 | 640 s |
| `cpu_weak_8` | 8 | 2048 | 8e6 | 680 s |

Slow degradation: +5% per 2× scale from Lustre contention, MPI startup,
synchronization barriers. At 8 nodes we expect ~13% overhead, which is
acceptable for a publication-grade scaling study.

---

## 3. GPU scaling — when and why

### 3.1 When to use GPU

SBI training is GPU-bound because:
- `posterior_nn("maf")` builds a stack of `num_transforms=4` MAF layers
  with `hidden_features=50`, applied to a `batch_size × D` tensor
- Each forward pass is `O(batch × D × hidden² × num_transforms)` matmul
- For batch=100, D=24, hidden=50: ~2.4M FLOPs per pass, run 100s of times per
  training step

For MC ground truth, **GPU does not help** because:
- The dominant cost is `GaussianFactorCopula.sample(n_scenarios=1e6)`,
  which is a sequential per-step normal-CDF transform
- The 1e6-scenario batch is ~1 GB — fits in 16 GB VRAM but the
  problem is memory-bandwidth bound, not compute bound
- Joblib on 256 CPU cores beats 1 MI250X GCD by ~3× for this workload

### 3.2 Multi-GPU scaling for SBI

**Strong scaling** (fixed N simulations, vary GPUs):
| Profile | GPUs | Expected time | Speedup vs 1 GPU |
|---|---|---|---|
| `gpu_single` | 1 | 1800 s (30 min) | 1.00× |
| `gpu_strong_4` | 4 (1 node) | 540 s | 3.33× |
| `gpu_multi_2` | 2 (2 nodes) | 950 s | 1.90× |
| `gpu_multi_4` | 4 (4 nodes) | 540 s | 3.33× |

Multi-node underperforms single-node DDP because:
- NCCL all-reduce over InfiniBand adds ~10–20% latency
- For small batches (B=100), the GPU compute is < communication
- Single-node 4-GPU has full NVLink-equivalent (XGMI on MI250X)

**Recommendation**: prefer `gpu_strong_4` over `gpu_multi_2/4` when
possible. The cost is the same (4 GPUs) but wall time is shorter.

---

## 4. Hybrid CPU+GPU — orchestrating concurrent workloads

The hybrid experiment runs two **independent** jobs in parallel:
- **GPU half**: SBI training (1 GPU, 32 CPU, 120 GB, 4h)
- **CPU half**: MC ground truth (128 CPU, 224 GB, 4h)

This is submitted as two separate jobs (not a job array, because Slurm
cannot mix partitions in an array). Both are independent — no
dependency. The CPU half uses posterior samples from a **prior** check-in
(`posterior_samples.npy`); the GPU half produces the same samples from
SBI training.

**Why hybrid**:
- Demonstrates that the LUMI scheduler can run our workloads in parallel
- Cuts wall time by ~50% vs sequential execution
- Real publication scenario: train posterior on GPU, immediately run MC
  on CPU cluster

**Resource budget for hybrid** (1 hour, 2 jobs in parallel):
- CPU: 128 CPU × 1h = 128 CPU-hr
- GPU: 1 GPU × 1h = 1 GPU-hr
- Total wall: 1h (parallel) vs ~3h (sequential)

---

## 5. What scales where — the bottom line

| Workload | Profile | Why |
|---|---|---|
| Smoke test | `cpu_small` (8 CPU, 30m) | Fast verification, no GPU contention |
| SBI NPE training | `gpu_single` (1 GPU, 4h) | Optimal GPU utilization for B=100 |
| SBI NLE/cMAF comparison | `gpu_single` ×3 | Each method independent |
| MC ground truth, dev | `cpu_med` (256 CPU, 1e6) | Quick iteration |
| MC ground truth, full | `cpu_large` (256 CPU, 1e7, 8h) | Publication-quality |
| MC weak scaling | `cpu_weak_2/4/8` | Publish scaling plot |
| MC strong scaling | `cpu_strong_8/16/32/64/128` | Publish Amdahl curve |
| Multi-GPU SBI | `gpu_strong_4` | Faster than multi-node |
| Hybrid (concurrent) | `hybrid_cpu` + `hybrid_gpu` | Wall-time reduction |

---

## 6. Job dependencies (DAG)

```
                   ┌─────────────┐
                   │  smoke_test │  (5 min)
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  sbi_train  │  (30 min, 1 GPU)
                   └──────┬──────┘
            ┌─────────────┼─────────────┐
            │             │             │
    ┌───────▼──────┐ ┌───▼─────┐ ┌─────▼───────┐
    │ cpu_strong_* │ │cpu_weak_*│ │ gpu_strong_4 │
    │  (job array) │ │ (4 jobs) │ │  + multi_2/4 │
    └───────┬──────┘ └───┬─────┘ └─────┬───────┘
            │            │              │
            └────────────┼──────────────┘
                         │
                  ┌──────▼──────┐
                  │   figures   │  (5 min)
                  └─────────────┘
```

Hybrid is submitted in parallel (no dependency on SBI):
```
    ┌────────────┐  ┌────────────┐
    │ hybrid_cpu │  │ hybrid_gpu │   (4h each, concurrent)
    └────────────┘  └────────────┘
```

---

## 7. The `scale_runner.py` abstraction

Each experiment script can call:

```python
from scale_runner import get_launcher
launcher = get_launcher()             # auto-detects Slurm env
results = launcher.map(work_fn, items)  # dispatches to joblib / DDP / sequential
```

The launcher auto-selects:
- `SequentialLauncher` — single process, no Slurm
- `JoblibLauncher` — multi-process within a node
- `SrunJoblibLauncher` — multi-node via srun + dask_jobqueue (optional)
- `TorchDDLauncher` — multi-GPU via `torch.distributed`

The right one is picked from `SLURM_NTASKS`, `SLURM_NNODES`, and
`CUDA_VISIBLE_DEVICES`. No experiment script has to know whether it's
running on 1 core or 256 nodes.

This means the **same** experiment script runs correctly:
- On a laptop (sequential)
- On `cpu_small` (joblib, 8 cores)
- On `cpu_weak_8` (joblib, 2048 cores)
- On `gpu_strong_4` (DDP, 4 GPUs)
- On `hybrid_cpu` (joblib, 128 cores)

---

## 8. Anti-patterns (what NOT to do)

- **Don't request GPUs for MC ground truth**: 1× MI250X = 1 GCD = $5/hr
  LUMI, and MC ground truth is ~3× slower on GPU than 256 CPU cores.
- **Don't request 1 huge node for 8+ hour runs**: Slurm preempts
  long jobs at 8h. Split into smaller chunks if you need > 8h.
- **Don't use 8 GPUs for SBI on K=10**: The forward simulator is the
  bottleneck, not the network. Stay at 1–4 GPUs.
- **Don't submit 20+ individual jobs**: Hit `AssocMaxSubmitJobLimit`.
  Use job arrays to pack 5+ strong-scaling points into 1 submission.
- **Don't `pip install` to home directory**: Home is over quota. Always
  `--target=./site-packages` in scratch.
