#!/usr/bin/env bash
# =============================================================================
# profiles/cpu_strong_128.sh — Strong scaling: 1 node, 128 CPUs
# =============================================================================
#SBATCH --job-name=qsvt4cra-cpu-strong-128
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-cpu_strong_128}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/dispatcher.sh"
