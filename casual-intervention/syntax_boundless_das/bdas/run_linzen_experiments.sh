#!/usr/bin/env bash
# Natural-text (Linzen) Boundless DAS: 2 models x 2 hypotheses x 2 directions.
# Usage: ./run_linzen_experiments.sh [seed] [gpu_id]
set -Euo pipefail

SEED="${1:-42}"
GPU_ID="${2:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_boundless_das.py"
LOG_DIR="$SCRIPT_DIR/logs_linzen/seed_${SEED}"
mkdir -p "$LOG_DIR"

CONFIGS=("config_linzen_qwen.yaml" "config_linzen_gemma.yaml")
DATASETS=("linzen_nat" "linzen_nat_2")            # H_syn, H_prox
DIRECTIONS=("source_to_base" "base_to_source")

RUN=0
TOTAL=$(( ${#CONFIGS[@]} * ${#DATASETS[@]} * ${#DIRECTIONS[@]} ))
echo "=== $TOTAL runs, seed $SEED, GPU $GPU_ID, started $(date) ==="

for cfg in "${CONFIGS[@]}"; do
    RUN_CONFIG="$(mktemp "$SCRIPT_DIR/.${cfg%.yaml}_seed_${SEED}.XXXXXX.yaml")"
    cp "$SCRIPT_DIR/$cfg" "$RUN_CONFIG"
    for dataset in "${DATASETS[@]}"; do
        for direction in "${DIRECTIONS[@]}"; do
            RUN=$((RUN + 1))
            sed -i -E "s/^[[:space:]]*dataset_name:.*/  dataset_name: $dataset/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*intervene_direction:.*/  intervene_direction: $direction/" "$RUN_CONFIG"
            sed -i -E "s/^[[:space:]]*seed:.*/  seed: $SEED/" "$RUN_CONFIG"

            TAG="${cfg%.yaml}__${dataset}__${direction}"
            echo "=========================================================="
            echo "Run $RUN/$TOTAL  $TAG  $(date)"
            echo "=========================================================="
            if python "$TRAIN_SCRIPT" --config "$RUN_CONFIG" 2>&1 | tee "$LOG_DIR/${TAG}.log"; then
                echo "[ok] run $RUN/$TOTAL $TAG"
            else
                echo "[FAILED] run $RUN/$TOTAL $TAG -- see $LOG_DIR/${TAG}.log"
            fi
        done
    done
    rm -f "$RUN_CONFIG"
done

echo "=== finished $(date) ==="
