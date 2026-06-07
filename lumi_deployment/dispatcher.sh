#!/usr/bin/env bash
# =============================================================================
# dispatcher.sh — Shared experiment dispatch logic for QSVT4CRA profiles
# -----------------------------------------------------------------------------
# Sourced by lumi_deployment/profiles/*.sh. Each profile is a complete Slurm
# script (with its own #SBATCH directives) that sources this file to run the
# actual experiment.
#
# Required environment variables (set by the profile or by the caller):
#   EXPERIMENT   : sbi_train | mc_ground_truth | mc_ground_truth_weak |
#                  qsvt_sweep | ood_robustness | resource_scaling |
#                  figures | hybrid_cpu_gpu | smoke_test
#   K, REGIME, COPULA, SEED, N_SCENARIOS, N_SIMULATIONS, N_ROUNDS, etc.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Module + scratch setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_lumi_env.sh"

mkdir -p "$SCRATCH/qsvt4cra-research/slurm_logs"
mkdir -p "$SCRATCH/qsvt4cra-research/results"
mkdir -p "$SCRATCH/qsvt4cra-research/checkpoints"
mkdir -p "$SCRATCH/qsvt4cra-research/figures"

cd "$SCRATCH/qsvt4cra-research"

EXPERIMENT="${EXPERIMENT:-sbi_train}"

echo "============================================================"
echo "QSVT4CRA-research-run job starting"
echo "  PROFILE      = ${PROFILE_NAME:-default}"
echo "  EXPERIMENT   = ${EXPERIMENT}"
echo "  Node         = $(hostname)"
echo "  SLURM_JOB_ID = ${SLURM_JOB_ID:-N/A}"
echo "  SLURM_NTASKS = ${SLURM_NTASKS:-1}"
echo "  SLURM_NNODES = ${SLURM_NNODES:-1}"
echo "  SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE:-1}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-(none)}"
echo "============================================================"

# Verify scale_runner can detect the environment
if [[ -f "scale_runner.py" ]]; then
    python -m scale_runner info 2>&1 | head -20 || true
fi

# ---------------------------------------------------------------------------
# 2. Experiment dispatch
# ---------------------------------------------------------------------------
case "$EXPERIMENT" in
  sbi_train)
      python -m experiments.sbi_train \
          --method "${SBI_METHOD:-npe}" \
          --K "${K:-10}" \
          --regime "${REGIME:-baseline}" \
          --n-simulations "${N_SIMULATIONS:-1000}" \
          --n-rounds "${N_ROUNDS:-10}" \
          --device "${SBI_DEVICE:-cuda}" \
          --seed "${SEED:-42}" \
          --output "checkpoints/sbi_${SBI_METHOD:-npe}_${REGIME:-baseline}_K${K:-10}.pt"
      ;;

  mc_ground_truth)
      # Requires a checkpoint from sbi_train
      CHECKPOINT="checkpoints/sbi_npe_${REGIME:-baseline}_K${K:-10}.pt"
      POSTERIOR_SAMPLES_FLAG=""
      if [[ -f "$CHECKPOINT" ]]; then
          POSTERIOR_SAMPLES_FLAG="--posterior-checkpoint $CHECKPOINT"
      elif [[ -f "posterior_samples.npy" ]]; then
          POSTERIOR_SAMPLES_FLAG="--posterior-samples posterior_samples.npy"
      else
          echo "ERROR: no checkpoint or posterior_samples.npy found" >&2
          echo "  Run EXPERIMENT=sbi_train first" >&2
          exit 1
      fi
      python -m experiments.mc_ground_truth \
          $POSTERIOR_SAMPLES_FLAG \
          --n-scenarios "${N_SCENARIOS:-1000000}" \
          --copula "${COPULA:-gaussian}" \
          --regime "${REGIME:-baseline}" \
          --K "${K:-10}" \
          --seed "${SEED:-42}" \
          --batch-size "${MC_BATCH_SIZE:-50000}" \
          --output "results/mc_ground_truth_${COPULA:-gaussian}_${REGIME:-baseline}_K${K:-10}.npz"
      ;;

  mc_ground_truth_weak)
      # Weak scaling: scenarios scale with node count
      NODES="${SLURM_NNODES:-1}"
      SCENARIOS=$((1000000 * NODES))
      echo "[weak-scaling] ${NODES} nodes -> ${SCENARIOS} scenarios (expected constant time)"
      # Use the .pt checkpoint from sbi_train (contains posterior_samples inside)
      CHECKPOINT="checkpoints/sbi_npe_baseline_K10.pt"
      if [[ -f "$CHECKPOINT" ]]; then
          POSTERIOR_FLAG="--posterior-checkpoint $CHECKPOINT"
      elif [[ -f "posterior_samples.npy" ]]; then
          POSTERIOR_FLAG="--posterior-samples posterior_samples.npy"
      else
          echo "ERROR: no posterior samples found (need sbi_train to run first)" >&2
          exit 1
      fi
      python -m experiments.mc_ground_truth \
          $POSTERIOR_FLAG \
          --n-scenarios "${SCENARIOS}" \
          --copula gaussian \
          --regime baseline \
          --K 10 \
          --batch-size 50000 \
          --output "results/mc_ground_truth_weak_${NODES}nodes.npz"
      ;;

  qsvt_sweep)
      python -m experiments.qsvt_sweep \
          --degrees 16 32 64 128 256 512 1024 \
          --posterior-checkpoint "checkpoints/sbi_npe_${REGIME:-baseline}_K${K:-10}.pt" \
          --output "results/qsvt_sweep_${REGIME:-baseline}.npz"
      ;;

  ood_robustness)
      CHECKPOINT="checkpoints/sbi_npe_baseline_K10.pt"
      if [[ -f "$CHECKPOINT" ]]; then
          POSTERIOR_FLAG="--posterior-checkpoint $CHECKPOINT"
      elif [[ -f "posterior_samples.npy" ]]; then
          POSTERIOR_FLAG="--posterior-samples posterior_samples.npy"
      else
          echo "ERROR: no posterior samples found" >&2
          exit 1
      fi
      python -m experiments.ood_robustness \
          $POSTERIOR_FLAG \
          --test-regimes baseline housing_crash rate_shock_0.5 rate_shock_1.5 unemployment liquidity \
          --n-posterior-samples "${N_POSTERIOR_SAMPLES:-1000}" \
          --K "${K:-10}" \
          --output "results/ood_robustness_K${K:-10}.npz"
      ;;

  resource_scaling)
      python -m experiments.resource_scaling \
          --n-loans 10 50 100 500 1000 \
          --degree 64 \
          --output "figures/resource_scaling_K10-1000.png" \
          --csv "results/resource_scaling.csv"
      ;;

  figures)
      python -m experiments.make_figures \
          --results-dir results/ \
          --output-dir figures/ \
          --K "${K:-10}" \
          --dpi 300
      ;;

  smoke_test)
      # 30-second smoke: verify environment, modules, packages
      echo "[smoke] Python: $(python --version 2>&1)"
      echo "[smoke] Torch: $(python -c 'import torch; print(torch.__version__, "CUDA=", torch.cuda.is_available())' 2>&1)"
      echo "[smoke] NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
      echo "[smoke] sbi: $(python -c 'import sbi; print(sbi.__version__)' 2>&1 | head -1)"
      echo "[smoke] qiskit: $(python -c 'import qiskit; print(qiskit.__version__)' 2>&1 | head -1)"
      echo "[smoke] joblib: $(python -c 'import joblib; print(joblib.__version__)' 2>&1 | head -1)"
      python -c "from scale_runner import detect_environment; import json; print(json.dumps(detect_environment(), indent=2))"
      echo "[smoke] All imports OK"
      ;;

  *)
      echo "Unknown EXPERIMENT=${EXPERIMENT}" >&2
      echo "Valid: sbi_train, mc_ground_truth, mc_ground_truth_weak, qsvt_sweep," >&2
      echo "       ood_robustness, resource_scaling, figures, smoke_test" >&2
      exit 1
      ;;
esac

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
