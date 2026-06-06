#!/usr/bin/env bash
# =============================================================================
# setup_lumi_env.sh
# -----------------------------------------------------------------------------
# Bootstrap a LUMI compute environment for QSVT4CRA-research-run experiments.
# Ported from JunctionHackathon / quantumhack lessons (2026-06).
#
# Usage:
#   bash setup_lumi_env.sh
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
export PYTHONPATH="${SCRATCH}/site-packages:${PYTHONPATH:-}"
export PYTHONUSERBASE="${SCRATCH}/python_user_base"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export HF_HOME="${SCRATCH}/huggingface"
export TORCH_HOME="${SCRATCH}/torch"

# ---------------------------------------------------------------------------
# 4. JAX / ROCm env (for any JAX-based SBI estimator)
# ---------------------------------------------------------------------------
export JAX_PLATFORMS=rocm
export JAX_ROCM_PLUGIN_PATH=""
# Note: install jax-rocm manually per https://docs.jax.dev/en/latest/jax_on_lumi.html

# ---------------------------------------------------------------------------
# 5. Sanity printout
# ---------------------------------------------------------------------------
echo "===================================================================="
echo "LUMI env ready"
echo "  SLURM_JOB_ACCOUNT = ${SLURM_JOB_ACCOUNT}"
echo "  SCRATCH           = ${SCRATCH}"
echo "  PYTHONPATH        = ${PYTHONPATH}"
echo "  python            = $(which python)"
echo "  python --version  = $(python --version)"
echo "  rocm              = $(module show rocm 2>&1 | grep -i version | head -1)"
echo "===================================================================="
