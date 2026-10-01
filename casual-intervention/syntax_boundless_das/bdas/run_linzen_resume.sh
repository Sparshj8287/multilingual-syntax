#!/usr/bin/env bash
# Resume the natural-text (Linzen) Boundless DAS sweep.
#
# Copy of run_linzen_experiments.sh with three additions, so the original stays
# untouched:
#   1. resume  -- a run whose *_all_nua_summary.json already exists is skipped,
#                 so the three runs that finished on 19-20 Sep are not repeated.
#   2. disk    -- refuse to start a run below MIN_FREE_GB instead of letting a
#                 full disk fail every queued run in seconds.
#   3. model   -- qwen | gemma | both, and gemma uses a reduced-batch config
#                 because batch 256 OOMed at 136 GiB on a full H200.
#
# Usage: ./run_linzen_resume.sh [seed] [gpu_id] [model]
set -Euo pipefail

SEED="${1:-42}"
GPU_ID="${2:-2}"
MODEL="${3:-both}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_boundless_das.py"
RESULT_ROOT="$SCRIPT_DIR/results_das_linzen_nat"
LOG_DIR="$SCRIPT_DIR/logs_linzen_resume/seed_${SEED}"
mkdir -p "$LOG_DIR"

case "$MODEL" in
    qwen)  CONFIGS=("config_linzen_qwen.yaml") ;;
    gemma) CONFIGS=("config_linzen_gemma_bs128.yaml") ;;
    both)  CONFIGS=("config_linzen_qwen.yaml" "config_linzen_gemma_bs128.yaml") ;;
    *) echo "model must be qwen, gemma or both (got '$MODEL')" >&2; exit 1 ;;
esac

DATASETS=("linzen_nat" "linzen_nat_2")            # H_syn, H_prox
DIRECTIONS=("source_to_base" "base_to_source")
VARIATION="linear"                                 # fixed in both configs

MIN_FREE_GB="${MIN_FREE_GB:-15}"
check_disk() {
    local free_gb
    free_gb=$(df -BG --output=avail "$SCRIPT_DIR" | tail -1 | tr -dc '0-9')
    if [[ -z "$free_gb" ]]; then
        echo "[disk] WARNING: could not read free space; continuing." >&2
        return 0
    fi
    if (( free_gb < MIN_FREE_GB )); then
        echo "=== STOPPING: only ${free_gb}GB free, below MIN_FREE_GB=${MIN_FREE_GB}GB ===" >&2
        return 1
    fi
    return 0
}

# A run is finished only once its all-NUA summary exists; per-layer result.json
# files can be left behind by an interrupted run.
summary_path() {
    local model_dir="$1" dataset="$2" direction="$3"
    echo "$RESULT_ROOT/$model_dir/$dataset/$VARIATION/seed_${SEED}/${direction}_all_nua_summary.json"
}

RUN=0
DONE=0
SKIPPED=0
FAILED=0
TOTAL=$(( ${#CONFIGS[@]} * ${#DATASETS[@]} * ${#DIRECTIONS[@]} ))
echo "=== $TOTAL candidate runs, model $MODEL, seed $SEED, GPU $GPU_ID, started $(date) ==="

for cfg in "${CONFIGS[@]}"; do
    case "$cfg" in
        *qwen*)  MODEL_DIR="qwen-3-8b" ;;
        *gemma*) MODEL_DIR="gemma-3-12b-pt" ;;
    esac
    RUN_CONFIG="$(mktemp "$SCRIPT_DIR/.${cfg%.yaml}_resume_${SEED}.XXXXXX.yaml")"
    cp "$SCRIPT_DIR/$cfg" "$RUN_CONFIG"
    for dataset in "${DATASETS[@]}"; do
        for direction in "${DIRECTIONS[@]}"; do
            RUN=$((RUN + 1))
            TAG="${cfg%.yaml}__${dataset}__${direction}"

            if [[ -f "$(summary_path "$MODEL_DIR" "$dataset" "$direction")" ]]; then
                SKIPPED=$((SKIPPED + 1))
                echo "[skip] $RUN/$TOTAL $TAG -- already complete"
                continue
            fi
            if ! check_disk; then
                rm -f "$RUN_CONFIG"
                echo "=== aborted at run $RUN/$TOTAL $(date) ==="
                exit 2
            fi

            sed -i -E "s/^[[:space:]]*dataset_name:.*/  dataset_name: $dataset/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*intervene_direction:.*/  intervene_direction: $direction/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*seed:.*/  seed: $SEED/" "$RUN_CONFIG"

            echo "=========================================================="
            echo "Run $RUN/$TOTAL  $TAG  $(date)"
            echo "=========================================================="
            if python "$TRAIN_SCRIPT" --config "$RUN_CONFIG" 2>&1 | tee "$LOG_DIR/${TAG}.log"; then
                DONE=$((DONE + 1))
                echo "[ok] run $RUN/$TOTAL $TAG"
            else
                FAILED=$((FAILED + 1))
                echo "[FAILED] run $RUN/$TOTAL $TAG -- see $LOG_DIR/${TAG}.log"
            fi
        done
    done
    rm -f "$RUN_CONFIG"
done

echo "=== finished $(date): $DONE ran, $SKIPPED skipped as complete, $FAILED failed ==="
