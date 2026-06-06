#!/usr/bin/env bash
# =============================================================================
# profiles/cpu_small.sh — Single-node CPU smoke test
# -----------------------------------------------------------------------------
# Submit directly:
#   EXPERIMENT=mc_ground_truth sbatch lumi_deployment/profiles/cpu_small.sh
# Or via the dispatcher:
#   PROFILE=cpu_small EXPERIMENT=mc_ground_truth \
#       bash lumi_deployment/slurm_qsvt4cra_research.sh
# =============================================================================
#SBATCH --job-name=qsvt4cra-cpu-small
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-cpu_small}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${SCRIPT_DIR}/dispatcher.sh"
