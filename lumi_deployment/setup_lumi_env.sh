#!/usr/bin/env bash
# =============================================================================
# setup_lumi_env.sh
# -----------------------------------------------------------------------------
# Bootstrap a LUMI compute environment for QSVT4CRA-research-run experiments.
# Ported from JunctionHackathon / quantumhack lessons (2026-06).
#
# Usage:
#   bash setup_lumi_env.sh
#   # or: source setup_lumi_env.sh    (so env vars persist in current shell)
#
# After this, prepend $SCRATCH/site-packages to PYTHONPATH and you are ready
# to import sbi, qiskit, torch, etc.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Module stack (verified 2026-06)
# ---------------------------------------------------------------------------
module --force purge
module load LUMI/25.09
module load partition/G
module load rocm                # rocm/6.4.4
module load cray-python/3.11.7

# ---------------------------------------------------------------------------
# 2. Scratch directory (avoid ~1 GB home quota)
# ---------------------------------------------------------------------------
: "${SLURM_JOB_ACCOUNT:=project_465003017}"
export SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}/${USER}"
mkdir -p "$SCRATCH"

# ---------------------------------------------------------------------------
# 3. PYTHONPATH for staged packages (populated by rsync_to_lumi.sh)
# ---------------------------------------------------------------------------
export PYTHONPATH="${SCRATCH}/qsvt4cra-research:${SCRATCH}/site-packages:${PYTHONPATH:-}"
export PYTHONUSERBASE="${SCRATCH}/python_user_base"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"
export WANDB_DIR="${SCRATCH}/wandb"
mkdir -p "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. JAX / ROCm env (for any JAX-based SBI estimator)
# ---------------------------------------------------------------------------
export JAX_PLATFORMS=rocm
export JAX_ROCM_PLUGIN_PATH=""
# Note: install jax-rocm manually per https://docs.jax.dev/en/latest/jax_on_lumi.html
# We do NOT ship jax-rocm by default; only enable if user has installed it.
if [[ -z "${JAX_PLATFORMS_FORCE:-}" ]]; then
    # Allow users to override (e.g., JAX_PLATFORMS_FORCE=cpu)
    if [[ -d "${SCRATCH}/site-packages/jax" ]]; then
        export JAX_PLATFORMS=rocm
    else
        unset JAX_PLATFORMS  # fall back to default
    fi
fi

# ---------------------------------------------------------------------------
# 5. Cray MPICH: use srun, not mpirun
# ---------------------------------------------------------------------------
# LUMI has no mpirun/mpiexec. We use srun for multi-node MPI.
# For Python multiprocessing, use srun -n $SLURM_NTASKS python ...
export USE_SRUN=1

# ---------------------------------------------------------------------------
# 6. Slurm: if we're inside a job, derive vars
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[lumi-env] Running inside Slurm job ${SLURM_JOB_ID}"
    echo "[lumi-env] Node list: ${SLURM_NODELIST}"
    echo "[lumi-env] CPUs/node: ${SLURM_CPUS_ON_NODE:-?}"
    echo "[lumi-env] GPUs/node: ${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-0}}"
fi

# ---------------------------------------------------------------------------
# 7. Sanity printout
# ---------------------------------------------------------------------------
echo "===================================================================="
echo "LUMI env ready"
echo "  SLURM_JOB_ACCOUNT = ${SLURM_JOB_ACCOUNT}"
echo "  SCRATCH           = ${SCRATCH}"
echo "  PYTHONPATH        = ${PYTHONPATH:0:100}..."
echo "  python            = $(which python)"
echo "  python --version  = $(python --version 2>&1)"
echo "  rocm              = $(module show rocm 2>&1 | grep -i version | head -1 || echo '(not loaded)')"
echo "===================================================================="
