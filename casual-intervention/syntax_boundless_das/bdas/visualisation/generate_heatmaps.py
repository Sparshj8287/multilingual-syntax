import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install seaborn and matplotlib: pip install seaborn matplotlib")
    exit(1)

results_dir = Path(__file__).resolve().parent.parent / "results"
output_dir = Path(__file__).resolve().parent / "plots"

output_dir.mkdir(exist_ok=True)

# data[group_name][column_name][layer] = accuracy
data = {}

# We recursively find all all_layers_summary.json files
for summary_path in results_dir.rglob("all_layers_summary.json"):
    # Expected path parts: results / {model} / {dataset} / {variation} / {nua}_attractor / {direction} / all_layers_summary.json
    parts = summary_path.relative_to(results_dir).parts
    if len(parts) < 6:
        continue
        
    model = parts[0]
    dataset = parts[1]
    variation = parts[2]
    nua_dir = parts[3]
    
    # Check if there is an extra config folder before direction
    if parts[4] not in ["source_to_base", "base_to_source"]:
        if len(parts) >= 7 and parts[5] in ["source_to_base", "base_to_source"]:
            config_name = parts[4]
            direction = parts[5]
            group_name = f"{dataset}_{variation}_{config_name}_{direction}"
        else:
            continue
    else:
        direction = parts[4]
        group_name = f"{dataset}_{variation}_{direction}"
    
    col_name = f"{model} ({nua_dir})"
    
    if group_name not in data:
        data[group_name] = {}
    if col_name not in data[group_name]:
        data[group_name][col_name] = {}
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        for layer_info in summary.get("layers", []):
            layer = layer_info.get("layer")
            accuracy = layer_info.get("final_test_accuracy", 0)
            data[group_name][col_name][layer] = accuracy

if not data:
    print(f"No summary files found in {results_dir}")
    exit(0)

for group_name, cols_data in data.items():
    df_data = {}
    all_layers = set()
    for col, layers_data in cols_data.items():
        all_layers.update(layers_data.keys())
    
    # Sort layers
    target_layers = sorted(list(all_layers))
    
    for col, layers_data in cols_data.items():
        full_layers = {l: np.nan for l in target_layers}
        for l in target_layers:
            if l in layers_data:
                full_layers[l] = layers_data[l]
        df_data[col] = full_layers
    
    if not df_data:
        continue
        
    df = pd.DataFrame(df_data)
    df.index.name = "Layer Index"
    
    # Custom sorting for columns so '1_attractor' comes before '10_attractor', etc.
    # It tries to extract the NUA integer and the model name for robust sorting.
    def sort_key(col_name):
        # e.g., "qwen-3-8b (1_attractor)"
        try:
            model_part, nua_part = col_name.rsplit(" (", 1)
            nua_val = int(nua_part.split("_")[0])
            return (model_part, nua_val)
        except Exception:
            return (col_name, 0)

    sorted_columns = sorted(df.columns, key=sort_key)
    df = df.reindex(sorted_columns, axis=1)
    
    # Sort index (layers) descending so highest layer is at top of heatmap
    df = df.sort_index(ascending=False)
    
    # Adjust width based on number of columns to ensure they are readable
    width = max(10, len(df.columns) * 1.5)
    plt.figure(figsize=(width, 8))
    
    yticklabels = [str(l) for l in df.index]
    annot_df = df.map(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
    
    sns.heatmap(df, annot=annot_df, fmt="", cmap="magma_r", cbar_kws={'label': 'Final Test Accuracy'}, yticklabels=yticklabels)
    plt.title(group_name)
    plt.xlabel("Model and Attractors")
    plt.ylabel("Layer Index")
    
    safe_filename = group_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{safe_filename}_heatmap.png"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Generated heatmap for {group_name} at {save_path}")
