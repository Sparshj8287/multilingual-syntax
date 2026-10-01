#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    echo "Usage: $0 <seed> [gpu_id]"
    echo "Example: $0 96 0"
    echo "You may also set CUDA_VISIBLE_DEVICES instead of passing gpu_id."
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

SEED="$1"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be a non-negative integer." >&2
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config_gemma.yaml"
TRAIN_SCRIPT="$SCRIPT_DIR/train_DBM.py"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

# An explicit gpu_id takes precedence, followed by the existing environment.
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

DATASETS=("obj_rel_across_anim" "obj_rel_across_anim_2")
VARIATIONS=("linear" "bow")
INTERVENTION_DIRECTIONS=("source_to_base" "base_to_source")

# Keep the source config unchanged. The temporary config remains in the same
# directory so that paths relative to config_olmo.yaml continue to work.
RUN_CONFIG="$(mktemp "$SCRIPT_DIR/.config_gemma_seed_${SEED}.XXXXXX.yaml")"
cp "$CONFIG" "$RUN_CONFIG"
trap 'rm -f "$RUN_CONFIG"' EXIT

TOTAL_RUNS=$((
    ${#DATASETS[@]}
    * ${#VARIATIONS[@]}
    * ${#INTERVENTION_DIRECTIONS[@]}
))
RUN_NUMBER=0

echo "Starting $TOTAL_RUNS Gemma experiments with seed $SEED on GPU(s): $CUDA_VISIBLE_DEVICES"

for dataset in "${DATASETS[@]}"; do
    for variation in "${VARIATIONS[@]}"; do
        for direction in "${INTERVENTION_DIRECTIONS[@]}"; do
            RUN_NUMBER=$((RUN_NUMBER + 1))

            sed -i -E "s/^[[:space:]]*dataset_name:.*/  dataset_name: $dataset/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*variation:.*/  variation: $variation/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*intervene_direction:.*/  intervene_direction: $direction/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*seed:.*/  seed: $SEED/" "$RUN_CONFIG"

            echo "=========================================================="
            echo "Run $RUN_NUMBER/$TOTAL_RUNS"
            echo "Dataset: $dataset"
            echo "Variation: $variation"
            echo "Intervention direction: $direction"
            echo "Seed: $SEED"
            echo "=========================================================="

            python "$TRAIN_SCRIPT" --config "$RUN_CONFIG"

            echo "Completed run $RUN_NUMBER/$TOTAL_RUNS"
        done
    done
done

echo "All $TOTAL_RUNS experiments finished successfully."
