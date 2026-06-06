#!/usr/bin/env bash
# =============================================================================
# profiles/gpu_multi_2.sh — 2-node multi-GPU training (DDP)
# =============================================================================
#SBATCH --job-name=qsvt4cra-gpu-multi-2
#SBATCH --account=project_465003017
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-gpu_multi_2}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${SCRIPT_DIR}/dispatcher.sh"
