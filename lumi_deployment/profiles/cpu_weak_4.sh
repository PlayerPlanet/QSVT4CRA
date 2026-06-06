#!/usr/bin/env bash
# =============================================================================
# profiles/cpu_weak_4.sh — Weak scaling: 4 nodes
# =============================================================================
#SBATCH --job-name=qsvt4cra-cpu-weak-4
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-cpu_weak_4}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/dispatcher.sh"
