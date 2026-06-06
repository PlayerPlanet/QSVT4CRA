#!/usr/bin/env bash
# =============================================================================
# profiles/gpu_multi_4.sh — 4-node multi-GPU training (DDP)
# =============================================================================
#SBATCH --job-name=qsvt4cra-gpu-multi-4
#SBATCH --account=project_465003017
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-gpu_multi_4}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/lumi_deployment"
source "${SCRIPT_DIR}/dispatcher.sh"
