#!/usr/bin/env bash
# =============================================================================
# profiles/hybrid_cpu.sh — Hybrid array task 0: CPU (MC ground truth)
# -----------------------------------------------------------------------------
# Half of the hybrid CPU+GPU experiment. Submitted as a job array; the
# dispatcher reads SLURM_ARRAY_TASK_ID to pick the right experiment.
# For hybrid, the dispatcher submits two SEPARATE jobs (one per array task
# line) with appropriate partition: see lumi_deployment/dispatch_hybrid.sh
# =============================================================================
#SBATCH --job-name=qsvt4cra-hybrid-cpu
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err
set -euo pipefail
PROFILE_NAME="${PROFILE_NAME:-hybrid_cpu}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Force the CPU half of the hybrid experiment
export EXPERIMENT="${EXPERIMENT:-mc_ground_truth}"
export COPUla="${COPULA:-gaussian}"
export REGIME="${REGIME:-baseline}"
export N_SCENARIOS="${N_SCENARIOS:-1000000}"
source "${SCRIPT_DIR}/dispatcher.sh"
