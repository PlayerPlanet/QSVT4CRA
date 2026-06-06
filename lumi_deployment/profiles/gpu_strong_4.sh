#!/usr/bin/env bash
# =============================================================================
# profiles/gpu_strong_4.sh — Single-node, 4 GPUs (DDP within node)
# =============================================================================
#SBATCH --job-name=qsvt4cra-gpu-strong-4
#SBATCH --account=project_465003017
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-gpu_strong_4}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/dispatcher.sh"
