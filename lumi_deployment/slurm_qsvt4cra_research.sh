#!/usr/bin/env bash
# =============================================================================
# slurm_qsvt4cra_research.sh
# -----------------------------------------------------------------------------
# Entry point that selects a Slurm profile and dispatches the experiment.
#
# This script is NOT a Slurm submission script (no #SBATCH directives).
# The actual Slurm submission target is lumi_deployment/profiles/<PROFILE>.sh.
#
# Usage:
#   # Standard pattern (preferred): submit the profile directly
#   sbatch lumi_deployment/profiles/cpu_med.sh
#   sbatch lumi_deployment/profiles/gpu_single.sh
#   sbatch lumi_deployment/profiles/cpu_weak_4.sh
#
#   # Or with this dispatcher (sets PROFILE for you):
#   PROFILE=cpu_med EXPERIMENT=mc_ground_truth \
#       bash lumi_deployment/slurm_qsvt4cra_research.sh
#
# Override any of the experiment parameters before submission:
#   PROFILE=gpu_single EXPERIMENT=sbi_train \
#       SBI_METHOD=nle N_SIMULATIONS=5000 \
#       bash lumi_deployment/slurm_qsvt4cra_research.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROFILE="${PROFILE:-cpu_med}"
PROFILE_PATH="${SCRIPT_DIR}/profiles/${PROFILE}.sh"

if [[ ! -f "$PROFILE_PATH" ]]; then
    echo "ERROR: profile not found: $PROFILE" >&2
    echo "  Looked in: $PROFILE_PATH" >&2
    echo "  Available profiles:" >&2
    ls -1 "${SCRIPT_DIR}/profiles/"*.sh 2>/dev/null | xargs -n1 basename | sed 's/\.sh$//' | sed 's/^/    /' >&2
    exit 1
fi

# Inject the profile name into the env for dispatcher.sh to use
export PROFILE_NAME="${PROFILE}"

echo "[dispatcher] PROFILE      = ${PROFILE}"
echo "[dispatcher] PROFILE_PATH = ${PROFILE_PATH}"
echo "[dispatcher] EXPERIMENT   = ${EXPERIMENT:-sbi_train}"
echo "[dispatcher] All env overrides honored by dispatcher.sh"

# Submit the profile directly to sbatch, with all current env vars
# inherited so the dispatcher can read EXPERIMENT, K, REGIME, etc.
exec sbatch \
    --export=ALL \
    "$PROFILE_PATH"
