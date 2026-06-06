#!/usr/bin/env bash
# =============================================================================
# dispatch_all.sh — Master orchestrator for the QSVT4CRA scaling matrix
# -----------------------------------------------------------------------------
# Submits the full experimental matrix to Slurm in dependency order:
#   1. Smoke test (1 node, 8 CPU)         : verify environment
#   2. SBI training (1 GPU, 1 node)       : produce posterior checkpoint
#   3. CPU strong scaling sweep (1 node)  : 8,16,32,64,128 CPUs
#   4. CPU weak scaling sweep (multi-node): 1,2,4,8 nodes
#   5. GPU strong scaling sweep           : 1,2,4 GPUs
#   6. Hybrid CPU+GPU                     : 2 concurrent jobs
#   7. Figure generation                  : depends on all results
#
# Total Slurm submissions: ~15 (well under AssocMaxSubmitJobLimit)
# Total compute estimate: 50-150 GPU-hr, 200-400 CPU-hr
#
# Usage (from LUMI login node):
#   bash lumi_deployment/dispatch_all.sh                # full matrix
#   bash lumi_deployment/dispatch_all.sh --smoke-only   # just the smoke test
#   bash lumi_deployment/dispatch_all.sh --skip-gpu      # CPU only
#   bash lumi_deployment/dispatch_all.sh --skip-weak     # no multi-node
#   bash lumi_deployment/dispatch_all.sh --dry-run       # show what would be submitted
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILES_DIR="${SCRIPT_DIR}/profiles"

# Argument parsing
DRY_RUN=false
SMOKE_ONLY=false
SKIP_GPU=false
SKIP_WEAK=false
SKIP_HYBRID=false
SKIP_STRONG=false
MAX_PARALLEL=4  # throttle concurrent submissions

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --smoke-only)   SMOKE_ONLY=true ;;
        --skip-gpu)     SKIP_GPU=true ;;
        --skip-weak)    SKIP_WEAK=true ;;
        --skip-hybrid)  SKIP_HYBRID=true ;;
        --skip-strong)  SKIP_STRONG=true ;;
        --max-parallel=*) MAX_PARALLEL="${arg#*=}" ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# \?//' | head -50
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg" >&2
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------
# Submit a profile and return its job ID
# ----------------------------------------------------------------------
submit_profile() {
    local profile="$1"
    local env_exports="$2"
    local profile_path="${PROFILES_DIR}/${profile}.sh"
    if [[ ! -f "$profile_path" ]]; then
        echo "ERROR: profile not found: $profile_path" >&2
        return 1
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] would submit: $profile (env: $env_exports)"
        echo "DRYRUN"
        return 0
    fi
    # Capture job ID from sbatch output
    local job_id
    job_id=$(sbatch \
        --export="$env_exports" \
        "$profile_path" 2>&1 | tee /dev/stderr | grep -oE '[0-9]+' | head -1)
    if [[ -z "$job_id" ]]; then
        echo "ERROR: sbatch failed for $profile" >&2
        return 1
    fi
    echo "$job_id"
}

# ----------------------------------------------------------------------
# Banner
# ----------------------------------------------------------------------
echo "============================================================"
echo "QSVT4CRA dispatch_all.sh — master orchestrator"
echo "============================================================"
echo "  Dry-run:       $DRY_RUN"
echo "  Smoke-only:    $SMOKE_ONLY"
echo "  Skip-GPU:      $SKIP_GPU"
echo "  Skip-Weak:     $SKIP_WEAK"
echo "  Skip-Hybrid:   $SKIP_HYBRID"
echo "  Skip-Strong:   $SKIP_STRONG"
echo "  Max-parallel:  $MAX_PARALLEL"
echo ""

# ----------------------------------------------------------------------
# Phase 0: Smoke test
# ----------------------------------------------------------------------
echo "=== Phase 0: Smoke test ==="
SMOKE_JOB=$(submit_profile cpu_small "ALL,EXPERIMENT=smoke_test")
if [[ "$SMOKE_JOB" == "DRYRUN" ]]; then :; else echo "  -> Job $SMOKE_JOB"; fi

if [[ "$SMOKE_ONLY" == "true" ]]; then
    echo "[done] smoke-only mode"
    exit 0
fi

# ----------------------------------------------------------------------
# Phase 1: SBI training (1 GPU)
# ----------------------------------------------------------------------
echo ""
echo "=== Phase 1: SBI training (1 GPU) ==="
SBI_DEPS="afterok:${SMOKE_JOB}"
SBI_JOB=$(submit_profile gpu_single "ALL,EXPERIMENT=sbi_train,SBI_METHOD=npe,N_SIMULATIONS=1000,N_ROUNDS=10,K=10,REGIME=baseline,SBI_DEVICE=cuda")
if [[ "$SBI_JOB" == "DRYRUN" ]]; then :; else echo "  -> Job $SBI_JOB (waits for smoke $SMOKE_JOB)"; fi

# ----------------------------------------------------------------------
# Phase 2: CPU strong scaling (1 node, varying CPUs)
# ----------------------------------------------------------------------
STRONG_JOBS=()
if [[ "$SKIP_STRONG" != "true" ]]; then
    echo ""
    echo "=== Phase 2: CPU strong scaling (1 node, 8/16/32/64/128 CPUs) ==="
    # Use job array to submit all 5 strong-scaling points in one submission
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] would submit job array: cpu_strong_{8,16,32,64,128}.sh"
    else
        # Write the array script to a temp file (process substitution is fragile on LUMI)
        STRONG_SCRIPT="$(mktemp -p "${TMPDIR:-/tmp}" qsvt4cra-strong-XXXXXX.sh)"
        cat > "$STRONG_SCRIPT" <<'STRONG_EOF'
#!/usr/bin/env bash
#SBATCH --job-name=qsvt4cra-cpu-strong
#SBATCH --account=project_465003017
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/cpu-strong-%A_%a.out
#SBATCH --error=/scratch/project_465003017/%u/qsvt4cra-research/slurm_logs/cpu-strong-%A_%a.err
set -euo pipefail
PROFILE_NAME="cpu_strong_${SLURM_ARRAY_TASK_ID}"
N_CPUS="${SLURM_ARRAY_TASK_ID}"
export EXPERIMENT="${EXPERIMENT:-mc_ground_truth_weak}"
export N_SCENARIOS="${N_SCENARIOS:-1000000}"
SCRIPT_DIR="/scratch/project_465003017/${USER}/qsvt4cra-research/lumi_deployment"
source "${SCRIPT_DIR}/dispatcher.sh"
STRONG_EOF
        chmod +x "$STRONG_SCRIPT"
        STRONG_JOB=$(sbatch --export=ALL \
            --array=8,16,32,64,128 \
            --dependency="afterok:${SBI_JOB}" \
            --parsable \
            "$STRONG_SCRIPT")
        rm -f "$STRONG_SCRIPT"
        echo "  -> Strong-scaling array job: $STRONG_JOB (waits for SBI $SBI_JOB)"
        STRONG_JOBS+=("$STRONG_JOB")
    fi
fi

# ----------------------------------------------------------------------
# Phase 3: CPU weak scaling (multi-node)
# ----------------------------------------------------------------------
WEAK_JOBS=()
if [[ "$SKIP_WEAK" != "true" ]]; then
    echo ""
    echo "=== Phase 3: CPU weak scaling (1, 2, 4, 8 nodes) ==="
    # Submit each weak-scaling point as a separate job (different node counts)
    # But to save submission slots, we use a job array
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] would submit job array: cpu_weak_{1,2,4,8}.sh"
    else
        for nn in 1 2 4 8; do
            WEAK_PROFILE="cpu_weak_${nn}"
            WEAK_JOB=$(sbatch --export=ALL \
                --dependency="afterok:${SBI_JOB}" \
                --parsable \
                "${PROFILES_DIR}/${WEAK_PROFILE}.sh")
            echo "  -> Weak ${nn} nodes: Job $WEAK_JOB"
            WEAK_JOBS+=("$WEAK_JOB")
        done
    fi
fi

# ----------------------------------------------------------------------
# Phase 4: GPU strong scaling
# ----------------------------------------------------------------------
GPU_JOBS=()
if [[ "$SKIP_GPU" != "true" ]]; then
    echo ""
    echo "=== Phase 4: GPU strong scaling (1, 2, 4 GPUs) ==="
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] would submit: gpu_single, gpu_strong_4, gpu_multi_2"
    else
        # Single GPU baseline
        GPU1_JOB=$(sbatch --export=ALL \
            --dependency="afterok:${SBI_JOB}" \
            --parsable \
            "${PROFILES_DIR}/gpu_single.sh")
        echo "  -> 1 GPU:  Job $GPU1_JOB"
        GPU_JOBS+=("$GPU1_JOB")

        # 4 GPUs on 1 node (DDP within node)
        GPU4_JOB=$(sbatch --export=ALL \
            --dependency="afterok:${SBI_JOB}" \
            --parsable \
            "${PROFILES_DIR}/gpu_strong_4.sh")
        echo "  -> 4 GPUs (1 node):  Job $GPU4_JOB"
        GPU_JOBS+=("$GPU4_JOB")

        # 2 nodes × 2 GPUs
        GPU2N_JOB=$(sbatch --export=ALL \
            --dependency="afterok:${SBI_JOB}" \
            --parsable \
            "${PROFILES_DIR}/gpu_multi_2.sh")
        echo "  -> 2 GPUs (2 nodes):  Job $GPU2N_JOB"
        GPU_JOBS+=("$GPU2N_JOB")
    fi
fi

# ----------------------------------------------------------------------
# Phase 5: Hybrid CPU + GPU
# ----------------------------------------------------------------------
if [[ "$SKIP_HYBRID" != "true" ]]; then
    echo ""
    echo "=== Phase 5: Hybrid CPU + GPU (concurrent) ==="
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[dry-run] would submit: hybrid_cpu + hybrid_gpu concurrently"
    else
        HYB_CPU_JOB=$(sbatch --export=ALL \
            --parsable \
            "${PROFILES_DIR}/hybrid_cpu.sh")
        echo "  -> Hybrid CPU:  Job $HYB_CPU_JOB"
        HYB_GPU_JOB=$(sbatch --export=ALL \
            --parsable \
            "${PROFILES_DIR}/hybrid_gpu.sh")
        echo "  -> Hybrid GPU:  Job $HYB_GPU_JOB"
    fi
fi

# ----------------------------------------------------------------------
# Phase 6: Figure generation (waits on all results)
# ----------------------------------------------------------------------
echo ""
echo "=== Phase 6: Figure generation (waits for all) ==="
ALL_DEPS=()
[[ -n "${SMOKE_JOB:-}" && "$SMOKE_JOB" != "DRYRUN" ]] && ALL_DEPS+=("$SMOKE_JOB")
[[ -n "${SBI_JOB:-}" && "$SBI_JOB" != "DRYRUN" ]] && ALL_DEPS+=("$SBI_JOB")
[[ ${#STRONG_JOBS[@]} -gt 0 ]] && ALL_DEPS+=("${STRONG_JOBS[@]}")
[[ ${#WEAK_JOBS[@]} -gt 0 ]] && ALL_DEPS+=("${WEAK_JOBS[@]}")
[[ ${#GPU_JOBS[@]} -gt 0 ]] && ALL_DEPS+=("${GPU_JOBS[@]}")

if [[ ${#ALL_DEPS[@]} -gt 0 ]]; then
    DEP_STR=$(IFS=:; echo "${ALL_DEPS[*]}")
    DEP_FLAG="afterok:${DEP_STR}"
else
    DEP_FLAG=""
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] would submit: figures (depends on: ${ALL_DEPS[*]:-none})"
else
    FIG_JOB=$(sbatch --export=ALL,EXPERIMENT=figures \
        $([[ -n "$DEP_FLAG" ]] && echo "--dependency=${DEP_FLAG}") \
        --parsable \
        "${PROFILES_DIR}/cpu_small.sh")
    echo "  -> Figures:  Job $FIG_JOB (depends on: ${ALL_DEPS[*]:-none})"
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Dispatch summary"
echo "============================================================"
echo "  Smoke test:    $SMOKE_JOB"
echo "  SBI train:     $SBI_JOB"
echo "  Strong CPUs:   ${STRONG_JOBS[*]:-none}"
echo "  Weak nodes:    ${WEAK_JOBS[*]:-none}"
echo "  GPU scaling:   ${GPU_JOBS[*]:-none}"
echo "  Figures:       ${FIG_JOB:-dry-run}"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem"
echo ""
echo "Results will be in /scratch/project_465003017/\$USER/qsvt4cra-research/"
