#!/usr/bin/env bash
# =============================================================================
# profiles/hybrid_gpu.sh — Hybrid array task 1: GPU (SBI training)
# =============================================================================
#SBATCH --job-name=qsvt4cra-hybrid-gpu
#SBATCH --account=project_465003017
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-hybrid_gpu}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/lumi_deployment"
# Force the GPU half of the hybrid experiment
export EXPERIMENT="${EXPERIMENT:-sbi_train}"
export SBI_METHOD="${SBI_METHOD:-npe}"
export K="${K:-10}"
export REGIME="${REGIME:-baseline}"
export N_SIMULATIONS="${N_SIMULATIONS:-1000}"
export N_ROUNDS="${N_ROUNDS:-10}"
export SBI_DEVICE="${SBI_DEVICE:-cuda}"
source "${SCRIPT_DIR}/dispatcher.sh"
