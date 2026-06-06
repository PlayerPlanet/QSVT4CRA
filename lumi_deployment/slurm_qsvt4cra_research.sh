#!/usr/bin/env bash
# =============================================================================
# slurm_qsvt4cra_research.sh
# -----------------------------------------------------------------------------
# Slurm wrapper for the QSVT4CRA-research-run experiments.
# Each phase (SBI training, MC ground truth, QSVT sweep, OOD, scaling) is
# selected by setting EXPERIMENT env var.
#
# Usage:
#   EXPERIMENT=sbi_train sbatch slurm_qsvt4cra_research.sh
#   EXPERIMENT=mc_ground_truth sbatch slurm_qsvt4cra_research.sh
#   EXPERIMENT=qsvt_sweep sbatch slurm_qsvt4cra_research.sh
#   EXPERIMENT=ood_robustness sbatch slurm_qsvt4cra_research.sh
#   EXPERIMENT=resource_scaling sbatch slurm_qsvt4cra_research.sh
# =============================================================================
#SBATCH --job-name=qsvt4cra-research
#SBATCH --account=project_465003017        # DO NOT use ${SLURM_JOB_ACCOUNT} - not expanded by sbatch
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Module + scratch setup
# ---------------------------------------------------------------------------
source /scratch/project_465003017/$USER/qsvt4cra-research/lumi_deployment/setup_lumi_env.sh

mkdir -p "$SCRATCH/qsvt4cra-research/slurm_logs"
mkdir -p "$SCRATCH/qsvt4cra-research/results"
mkdir -p "$SCRATCH/qsvt4cra-research/checkpoints"
mkdir -p "$SCRATCH/qsvt4cra-research/figures"

cd "$SCRATCH/qsvt4cra-research"

EXPERIMENT="${EXPERIMENT:-sbi_train}"
echo "Running EXPERIMENT=${EXPERIMENT}"
echo "Node: $(hostname)"
echo "GPUs: $(rocm-smi --showproductname 2>/dev/null | head -20)"

# ---------------------------------------------------------------------------
# 1. Experiment dispatch
# ---------------------------------------------------------------------------
case "$EXPERIMENT" in
  sbi_train)
      python -m experiments.sbi_train \
          --config configs/sbi_npe_factocopula.yaml \
          --output checkpoints/sbi_npe_factocopula.pt
      ;;

  mc_ground_truth)
      python -m experiments.mc_ground_truth \
          --n-scenarios 10000000 \
          --posterior-checkpoint checkpoints/sbi_npe_factocopula.pt \
          --output results/mc_ground_truth.npz
      ;;

  qsvt_sweep)
      python -m experiments.qsvt_sweep \
          --degrees 16 32 64 128 256 512 1024 \
          --posterior-checkpoint checkpoints/sbi_npe_factocopula.pt \
          --output results/qsvt_sweep.npz
      ;;

  ood_robustness)
      python -m experiments.ood_robustness \
          --posterior-checkpoint checkpoints/sbi_npe_factocopula.pt \
          --regimes low_rate high_rate housing_crash \
          --output results/ood_robustness.npz
      ;;

  resource_scaling)
      python -m experiments.resource_scaling \
          --n-loans 10 50 100 500 1000 \
          --output results/resource_scaling.npz
      ;;

  figures)
      python -m experiments.make_figures \
          --results-dir results/ \
          --output-dir figures/
      ;;

  *)
      echo "Unknown EXPERIMENT=${EXPERIMENT}"
      echo "Valid: sbi_train, mc_ground_truth, qsvt_sweep, ood_robustness, resource_scaling, figures"
      exit 1
      ;;
esac

echo "Done: $(date)"
