#!/bin/bash

# =========================================================================
# Run this script with a specific GPU, for example:
# CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh
#
# To run on the second GPU, modify the CONFIGS array to contain the other
# two models (e.g., config_olmo.yaml and config_qwen.yaml) and run:
# CUDA_VISIBLE_DEVICES=1 ./run_experiments.sh
# =========================================================================

# List of config files for the models to run on this GPU
CONFIGS=("config_qwen.yaml" "config_olmo.yaml")

# Datasets and variations to iterate over
DATASETS=("obj_rel_across_anim_2" "obj_rel_across_anim")
VARIATIONS=("linear" "bow")

# Intervention type to use
INTERVENTION="base_to_source"

for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo "Error: Config file $config not found!"
        continue
    fi

    for dataset in "${DATASETS[@]}"; do
        for variation in "${VARIATIONS[@]}"; do
            echo "=========================================================="
            echo "Running Model Config: $config"
            echo "Dataset: $dataset | Variation: $variation"
            echo "Intervention: $INTERVENTION"
            echo "=========================================================="
            
            # Update the YAML config file using sed (handles leading spaces)
            sed -i -E "s/^[[:space:]]*dataset_name:.*/  dataset_name: $dataset/" "$config"
            sed -i -E "s/^[[:space:]]*variation:.*/  variation: $variation/" "$config"
            sed -i -E "s/^[[:space:]]*intervene_direction:.*/  intervene_direction: $INTERVENTION/" "$config"
            
            # Execute the training script
            python train_DBM.py --config "$config"
            
            echo "----------------------------------------------------------"
            echo "Completed $config for $dataset ($variation)"
            echo "----------------------------------------------------------"
            echo ""
        done
    done
done

echo "All experiments finished successfully!"
