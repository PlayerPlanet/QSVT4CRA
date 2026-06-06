#!/usr/bin/env bash
# =============================================================================
# slurm_heron_simulation.sh
# -----------------------------------------------------------------------------
# LUMI wrapper for IBM Heron r3 / ibm_boston calibration-aware Aer simulation.
#
# Usage from synced repo on LUMI:
#   export IBM_API_KEY=...       # do not echo this value
#   sbatch --export=ALL lumi_deployment/slurm_heron_simulation.sh
#
# Optional overrides:
#   BACKEND_NAME=ibm_boston DEGREE=4 SHOTS=1024 TARGET_LOSS_FRACTION=0.5 \
#     sbatch --export=ALL lumi_deployment/slurm_heron_simulation.sh
# =============================================================================
#SBATCH --job-name=qsvt4cra-heron-sim
#SBATCH --account=project_465003017
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err

set -euo pipefail

source /scratch/project_465003017/$USER/qsvt4cra-research/lumi_deployment/setup_lumi_env.sh

mkdir -p "$SCRATCH/qsvt4cra-research/slurm_logs"
mkdir -p "$SCRATCH/qsvt4cra-research/results"

cd "$SCRATCH/qsvt4cra-research"

: "${BACKEND_NAME:=ibm_boston}"
: "${DEGREE:=4}"
: "${SHOTS:=1024}"
: "${TARGET_LOSS_FRACTION:=0.5}"
: "${OUTPUT:=results/heron_k17_${BACKEND_NAME}_d${DEGREE}.json}"

if [[ -z "${IBM_API_KEY:-}" ]]; then
  echo "ERROR: IBM_API_KEY is not set. Export it before sbatch, e.g. sbatch --export=ALL ..." >&2
  exit 2
fi

echo "===================================================================="
echo "QSVT4CRA Heron simulation"
echo "  backend              = ${BACKEND_NAME}"
echo "  degree               = ${DEGREE}"
echo "  shots                = ${SHOTS}"
echo "  target_loss_fraction = ${TARGET_LOSS_FRACTION}"
echo "  output               = ${OUTPUT}"
echo "  node                 = $(hostname)"
echo "===================================================================="

python -m experiments.heron_simulation \
  --backend-name "${BACKEND_NAME}" \
  --degree "${DEGREE}" \
  --shots "${SHOTS}" \
  --target-loss-fraction "${TARGET_LOSS_FRACTION}" \
  --output "${OUTPUT}" \
  --api-key-env IBM_API_KEY

echo "Done: $(date)"
