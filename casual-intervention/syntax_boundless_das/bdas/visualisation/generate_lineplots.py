import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import seaborn as sns

results_dir = Path(__file__).resolve().parent.parent / "results"
output_dir = Path(__file__).resolve().parent / "plots_lines"

output_dir.mkdir(exist_ok=True)

# Set publication style
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# data[model][method][dataset][variation][attractor][layer] = accuracy
data = {}

for summary_path in results_dir.rglob("all_layers_summary.json"):
    parts = summary_path.relative_to(results_dir).parts
    if len(parts) < 6:
        continue
        
    model = parts[0]
    dataset = parts[1]
    variation = parts[2]
    nua_dir = parts[3]
    
    if parts[4] not in ["source_to_base", "base_to_source"]:
        if len(parts) >= 7 and parts[5] in ["source_to_base", "base_to_source"]:
            # config_name = parts[4]
            direction = parts[5]
        else:
            continue
    else:
        direction = parts[4]
        
    method = direction
    
    try:
        attractor = int(nua_dir.split('_')[0])
    except ValueError:
        continue

    if model not in data:
        data[model] = {}
    if method not in data[model]:
        data[model][method] = {}
    if dataset not in data[model][method]:
        data[model][method][dataset] = {}
    if variation not in data[model][method][dataset]:
        data[model][method][dataset][variation] = {}
    if attractor not in data[model][method][dataset][variation]:
        data[model][method][dataset][variation][attractor] = {}
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        for layer_info in summary.get("layers", []):
            layer = layer_info.get("layer")
            accuracy = layer_info.get("final_test_accuracy", 0)
            data[model][method][dataset][variation][attractor][layer] = accuracy

if not data:
    print(f"No summary files found in {results_dir}")
    exit(0)

attractor_styles = {
    1: {'linestyle': '-', 'marker': 'o', 'color': 'tab:blue', 'label': '1 Attractor'},
    2: {'linestyle': '--', 'marker': 's', 'color': 'tab:green', 'label': '2 Attractors'},
    3: {'linestyle': ':', 'marker': '^', 'color': 'tab:red', 'label': '3 Attractors'},
    4: {'linestyle': '-.', 'marker': 'D', 'color': 'tab:purple', 'label': '4 Attractors'}
}

# The 2x2 grid layout requirements:
# Left Upper: linear from obj_rel_across_anim_2
# Right Upper: linear from obj_rel_across_anim
# Left Lower: bow from obj_rel_across_anim_2
# Right Lower: bow from obj_rel_across_anim

grid_layout = [
    [("obj_rel_across_anim_2", "linear"), ("obj_rel_across_anim", "linear")],
    [("obj_rel_across_anim_2", "bow"), ("obj_rel_across_anim", "bow")]
]

for model, methods in data.items():
    for method, datasets in methods.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Model: {model} | Direction: {method}", fontsize=20, fontweight='bold', y=1.02)
        
        for row in range(2):
            for col in range(2):
                dataset, variation = grid_layout[row][col]
                ax = axes[row, col]
                
                # Check if data exists
                if dataset in datasets and variation in datasets[dataset]:
                    attractor_data = datasets[dataset][variation]
                    
                    # Get all unique layers across attractors to set X axis
                    all_layers = set()
                    for attr, layers in attractor_data.items():
                        all_layers.update(layers.keys())
                    sorted_layers = sorted(list(all_layers))
                    
                    for attr in sorted(attractor_data.keys()):
                        if attr in attractor_styles:
                            layers_dict = attractor_data[attr]
                            # Extract accuracy for sorted layers, use NaN if missing
                            accuracies = [layers_dict.get(l, np.nan) for l in sorted_layers]
                            
                            style = attractor_styles[attr]
                            ax.plot(sorted_layers, accuracies, 
                                    linestyle=style['linestyle'],
                                    marker=style['marker'],
                                    color=style['color'],
                                    label=style['label'])
                
                title_ds = dataset.replace('_', ' ').title().replace('Obj Rel', 'Object Relative')
                title_var = variation.title()
                ax.set_title(f"{title_ds} ({title_var})")
                ax.set_xlabel("Layer Index")
                ax.set_ylabel("Accuracy")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
        plt.tight_layout()
        filename = f"{model}_{method}_das_combined.pdf"
        save_path = output_dir / filename
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Generated PDF for {model} - {method} at {save_path}")
