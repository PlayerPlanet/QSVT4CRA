#!/usr/bin/env bash
# =============================================================================
# rsync_to_lumi.sh
# -----------------------------------------------------------------------------
# Sync local source + dependencies to LUMI scratch, working around the
# home-quota and cross-FS rename gotchas.
#
# Usage:
#   bash rsync_to_lumi.sh                         # sync default paths
#   REMOTE=kkiirikk@lumi.csc.fi bash rsync_to_lumi.sh
#   ACCOUNT=project_465003017 bash rsync_to_lumi.sh
#
# Requires: ssh config alias 'lumi' (or override REMOTE).
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:-kkiirikk@lumi.csc.fi}"
ACCOUNT="${ACCOUNT:-project_465003017}"
SCRATCH_REMOTE="/scratch/${ACCOUNT}/\${USER}/qsvt4cra-research"

# Local paths
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_SRC="${LOCAL_ROOT}/Code"
LOCAL_RUN="${LOCAL_ROOT}/lumi_deployment"
LOCAL_REQ="${LOCAL_ROOT}/requirements.txt"
LOCAL_PKGS="${LOCAL_ROOT}"

# Project subdirs that contain code we want on LUMI
PROJECT_DIRS=(
    "compiler_backend"
    "copula"
    "data"
    "loader"
    "metrics"
    "qsvt"
    "sbi_pipeline"
    "simulator"
    "experiments"
    "tests"
    "lumi_deployment"
)

# Top-level project files
TOP_LEVEL_FILES=(
    "scale_runner.py"
    "pytest.ini"
    "requirements.txt"
)

echo "[1/4] Creating remote directory structure..."
ssh "$REMOTE" "mkdir -p ${SCRATCH_REMOTE}/{Code,site-packages,results,checkpoints,figures,slurm_logs}"
echo ""

echo "[2/4] Syncing top-level files..."
for f in "${TOP_LEVEL_FILES[@]}"; do
    if [[ -f "${LOCAL_PKGS}/${f}" ]]; then
        echo "  rsync ${f}"
        rsync -av --no-relative "${LOCAL_PKGS}/${f}" "${REMOTE}:${SCRATCH_REMOTE}/"
    else
        echo "  skip ${f} (not found)"
    fi
done
echo ""

echo "[3/4] Syncing project source directories..."
for d in "${PROJECT_DIRS[@]}"; do
    if [[ -d "${LOCAL_PKGS}/${d}" ]]; then
        echo "  rsync ${d}/"
        rsync -av --delete \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            "${LOCAL_PKGS}/${d}/" "${REMOTE}:${SCRATCH_REMOTE}/${d}/"
    else
        echo "  skip ${d}/ (not found)"
    fi
done
echo ""

echo "[4/4] Syncing Code/ (paper repo, must stay untouched)..."
if [[ -d "${LOCAL_SRC}" ]]; then
    rsync -av --delete \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "${LOCAL_SRC}/" "${REMOTE}:${SCRATCH_REMOTE}/Code/"
fi
echo ""

echo "Done. Next:"
echo "  ssh ${REMOTE}"
echo "  cd ${SCRATCH_REMOTE}"
echo "  bash lumi_deployment/setup_lumi_env.sh"
echo "  pip install --no-cache-dir --target=./site-packages -r requirements.txt"
