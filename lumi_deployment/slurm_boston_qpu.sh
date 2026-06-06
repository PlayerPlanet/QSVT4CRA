#!/usr/bin/env bash
# =============================================================================
# slurm_boston_qpu.sh
# -----------------------------------------------------------------------------
# LUMI wrapper that compiles the K=17 Finnish mortgage QSVT circuit, submits
# it to the real IBM ``ibm_boston`` (Heron r3) QPU, and benchmarks the
# measurement against classical VaR/CVaR ground truth.
#
# Usage from synced repo on LUMI:
#   export IBM_API_KEY=...       # do not echo this value
#   sbatch --export=ALL lumi_deployment/slurm_boston_qpu.sh
#
# Optional overrides:
#   BACKEND_NAME=ibm_boston DEGREE=4 SHOTS=4096 TARGET_LOSS_FRACTION=0.5 \
#     N_CLASSICAL_SCENARIOS=200000 \
#     sbatch --export=ALL lumi_deployment/slurm_boston_qpu.sh
#
# Token resolution (in order): $IBM_API_KEY -> $IBM_API_KEY_FILE -> .ibm_token
# in the working directory.  See compiler_backend.heron.load_ibm_token.
# =============================================================================
#SBATCH --job-name=qsvt4cra-boston-qpu
#SBATCH --account=project_465003017
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --mem=64G
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
: "${SHOTS:=4096}"
: "${TARGET_LOSS_FRACTION:=0.5}"
: "${N_CLASSICAL_SCENARIOS:=200000}"
: "${OUTPUT:=results/boston_qpu_k17_${BACKEND_NAME}_d${DEGREE}.json}"
: "${IBM_API_KEY_FILE:=}"

if [[ -z "${IBM_API_KEY:-}" && -z "${IBM_API_KEY_FILE:-}" && ! -f .ibm_token ]]; then
  echo "ERROR: No IBM Quantum token found. Set \$IBM_API_KEY, set \$IBM_API_KEY_FILE, or place a .ibm_token file in the working directory." >&2
  exit 2
fi

echo "===================================================================="
echo "QSVT4CRA Boston QPU benchmark"
echo "  backend               = ${BACKEND_NAME}"
echo "  degree                = ${DEGREE}"
echo "  shots                 = ${SHOTS}"
echo "  target_loss_fraction  = ${TARGET_LOSS_FRACTION}"
echo "  n_classical_scenarios = ${N_CLASSICAL_SCENARIOS}"
echo "  output                = ${OUTPUT}"
echo "  node                  = $(hostname)"
echo "===================================================================="

python -m experiments.boston_qpu \
  --backend-name "${BACKEND_NAME}" \
  --degree "${DEGREE}" \
  --shots "${SHOTS}" \
  --target-loss-fraction "${TARGET_LOSS_FRACTION}" \
  --n-classical-scenarios "${N_CLASSICAL_SCENARIOS}" \
  --output "${OUTPUT}" \
  --api-key-env IBM_API_KEY

echo "Done: $(date)"
