#!/bin/bash

# Usage: bash run_layers.sh [config_file] [start_layer] [end_layer] [mode] [extra args...]
# Example: bash run_layers.sh configs/gemma/3-1b/english.yaml 0 25
# Example (multilingual): bash run_layers.sh configs/gemma/3-1b/structural/base.yaml 0 25 multilang --experiments in_lang,holdout

CONFIG_FILE=$1
START_LAYER=$2
END_LAYER=$3
MODE=$4
EXTRA_ARGS="${@:5}"

if [ -z "$CONFIG_FILE" ] || [ -z "$START_LAYER" ] || [ -z "$END_LAYER" ]; then
    echo "Usage: bash run_layers.sh [config_file] [start_layer] [end_layer]"
    echo "Example: bash run_layers.sh configs/gemma/3-1b/english.yaml 0 25"
    exit 1
fi

for (( layer=$START_LAYER; layer<=$END_LAYER; layer++ ))
do
    echo "========================================================"
    echo "Running probe on Layer $layer"
    echo "========================================================"

    # Create a temporary config for this layer
    TEMP_CONFIG="temp_config_layer_${layer}.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"

    # Update both model.model_layer and decoder_model.layer
    
    # 1. Update model.model_layer
    sed -i "s/model_layer: [0-45]*/model_layer: $layer/" "$TEMP_CONFIG"
    
    # 2. Update decoder_model.layer
    sed -i "s/  layer: [0-45]*/  layer: $layer/" "$TEMP_CONFIG"

    # Modify the output root directory to include the layer number
    # This prevents all layers from dumping into the same parent folder
    # Read the original root from the config (e.g., experiments/llama-3p1-1b/en)
    if [ "$MODE" = "multilang" ]; then
        python3 probing/run_multilingual_experiments.py --config "$TEMP_CONFIG" $EXTRA_ARGS
    else
        python3 probing/run_experiment.py "$TEMP_CONFIG"
    fi

    # Cleanup
    rm "$TEMP_CONFIG"
    
    echo "Finished Layer $layer"
    echo ""
done

echo "All layers $START_LAYER to $END_LAYER completed."
