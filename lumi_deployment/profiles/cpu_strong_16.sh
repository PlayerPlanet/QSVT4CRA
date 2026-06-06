#!/usr/bin/env bash
# =============================================================================
# profiles/cpu_strong_16.sh — Strong scaling: 1 node, 16 CPUs
# =============================================================================
#SBATCH --job-name=qsvt4cra-cpu-strong-16
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-cpu_strong_16}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/lumi_deployment"
source "${SCRIPT_DIR}/dispatcher.sh"
