#!/usr/bin/env bash
# Experiment 3: position-resolved Boundless DAS at NUA=4.
#   2 models x 2 positions (p0, p4) x 2 datasets x 2 variations x 2 directions = 32 runs.
# The dataset x variation grid covers all three hypotheses:
#   obj_rel_across_anim   / linear -> H_syn   (proximity-isolated split)
#   obj_rel_across_anim   / bow    -> H_syn   (majority-isolated split)
#   obj_rel_across_anim_2 / linear -> H_prox
#   obj_rel_across_anim_2 / bow    -> H_maj
# Usage: ./run_position_experiments.sh [seed] [gpu_id] [model]
#   model: gemma | qwen | both (default) -- lets the two halves run on separate GPUs.
set -Euo pipefail

SEED="${1:-42}"
GPU_ID="${2:-2}"
MODEL="${3:-both}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_boundless_das_pos.py"
LOG_DIR="$SCRIPT_DIR/logs_positions/seed_${SEED}"
mkdir -p "$LOG_DIR"

case "$MODEL" in
    gemma) CONFIGS=("config_pos_gemma.yaml") ;;
    qwen)  CONFIGS=("config_pos_qwen.yaml") ;;
    both)  CONFIGS=("config_pos_gemma.yaml" "config_pos_qwen.yaml") ;;
    *) echo "model must be gemma, qwen or both (got '$MODEL')" >&2; exit 1 ;;
esac
POSITIONS=("p0" "p4")
DATASETS=("obj_rel_across_anim" "obj_rel_across_anim_2")
VARIATIONS=("linear" "bow")
DIRECTIONS=("source_to_base" "base_to_source")

# A full disk previously failed every queued run in seconds instead of stopping,
# so refuse to start another run without room for its checkpoints (~130 MB per
# layer, plus logs and result files).
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
        echo "=== Completed runs are intact. Free space and relaunch to continue. ===" >&2
        return 1
    fi
    echo "[disk] ${free_gb}GB free"
    return 0
}

RUN=0
TOTAL=$(( ${#CONFIGS[@]} * ${#POSITIONS[@]} * ${#DATASETS[@]} * ${#VARIATIONS[@]} * ${#DIRECTIONS[@]} ))
FAILED=0
echo "=== $TOTAL runs, model $MODEL, seed $SEED, GPU $GPU_ID, started $(date) ==="

for cfg in "${CONFIGS[@]}"; do
    RUN_CONFIG="$(mktemp "$SCRIPT_DIR/.${cfg%.yaml}_seed_${SEED}.XXXXXX.yaml")"
    cp "$SCRIPT_DIR/$cfg" "$RUN_CONFIG"
    for position in "${POSITIONS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for variation in "${VARIATIONS[@]}"; do
                for direction in "${DIRECTIONS[@]}"; do
                    RUN=$((RUN + 1))
                    if ! check_disk; then
                        rm -f "$RUN_CONFIG"
                        echo "=== aborted at run $RUN/$TOTAL $(date) ==="
                        exit 2
                    fi
                    sed -i -E "s/^[[:space:]]*position:.*/  position: $position/" "$RUN_CONFIG"
                    sed -i -E "s/^[[:space:]]*dataset_name:.*/  dataset_name: $dataset/" "$RUN_CONFIG"
                    sed -i -E "s/^[[:space:]]*variation:.*/  variation: $variation/" "$RUN_CONFIG"
                    sed -i -E "s/^[[:space:]]*intervene_direction:.*/  intervene_direction: $direction/" "$RUN_CONFIG"
                    sed -i -E "s/^[[:space:]]*seed:.*/  seed: $SEED/" "$RUN_CONFIG"

                    TAG="${cfg%.yaml}__${position}__${dataset}__${variation}__${direction}"
                    echo "=========================================================="
                    echo "Run $RUN/$TOTAL  $TAG  $(date)"
                    echo "=========================================================="
                    if python "$TRAIN_SCRIPT" --config "$RUN_CONFIG" 2>&1 | tee "$LOG_DIR/${TAG}.log"; then
                        echo "[ok] run $RUN/$TOTAL $TAG"
                    else
                        FAILED=$((FAILED + 1))
                        echo "[FAILED] run $RUN/$TOTAL $TAG -- see $LOG_DIR/${TAG}.log"
                    fi
                done
            done
        done
    done
    rm -f "$RUN_CONFIG"
done

echo "=== finished $(date): $((TOTAL - FAILED))/$TOTAL succeeded, $FAILED failed ==="
