#!/usr/bin/env bash
# =============================================================================
# rsync_to_lumi.sh
# -----------------------------------------------------------------------------
# Sync local source + dependencies to LUMI scratch, working around the
# home-quota and cross-FS rename gotchas.
#
# Usage:
#   bash rsync_to_lumi.sh                # sync default paths
#   REMOTE=kkiirikk@lumi.csc.fi bash rsync_to_lumi.sh
#
# Requires: ssh config alias 'lumi' (or override REMOTE).
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:-lumi}"
ACCOUNT="${ACCOUNT:-project_465003017}"
SCRATCH_REMOTE="/scratch/${ACCOUNT}/\${USER}/qsvt4cra-research"

# Local paths
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_SRC="${LOCAL_ROOT}/Code"
LOCAL_REQ="${LOCAL_ROOT}/requirements.txt"
LOCAL_RUN="${LOCAL_ROOT}/lumi_deployment"

echo "[1/3] Syncing source -> ${REMOTE}:${SCRATCH_REMOTE}/Code/"
ssh "$REMOTE" "mkdir -p ${SCRATCH_REMOTE}/Code ${SCRATCH_REMOTE}/site-packages"
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    "${LOCAL_SRC}/" "${REMOTE}:${SCRATCH_REMOTE}/Code/"

echo "[2/3] Syncing lumi_deployment/ -> ${REMOTE}:${SCRATCH_REMOTE}/"
rsync -av "${LOCAL_RUN}/" "${REMOTE}:${SCRATCH_REMOTE}/lumi_deployment/"

echo "[3/3] Syncing requirements.txt"
rsync -av "${LOCAL_REQ}" "${REMOTE}:${SCRATCH_REMOTE}/requirements.txt"

echo "Done. Next:"
echo "  ssh ${REMOTE}"
echo "  cd ${SCRATCH_REMOTE}"
echo "  bash lumi_deployment/setup_lumi_env.sh"
echo "  pip install --no-cache-dir --target=./site-packages -r requirements.txt"
