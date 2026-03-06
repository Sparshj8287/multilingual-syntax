import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plots"))
max_layer = 45

os.makedirs(output_dir, exist_ok=True)

datasets = ["simple_agreement", "obj_rel_across_anim"]

data = {}

models = [m for m in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, m))]

for model in models:
    model_path = os.path.join(results_dir, model)
    for dataset in datasets:
        dataset_path = os.path.join(model_path, dataset)
        if not os.path.exists(dataset_path):
            continue
            
        configs = [c for c in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, c)) 
                   and c not in ["source_to_base", "base_to_source"]]
        
        if not configs:
            summary_path = os.path.join(dataset_path, "source_to_base", "all_layers_summary.json")
            if os.path.exists(summary_path):
                dataset_config_name = dataset
                if dataset_config_name not in data:
                    data[dataset_config_name] = {}
                data[dataset_config_name][model] = {}
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    for layer_info in summary.get("layers", []):
                        layer = layer_info.get("layer")
                        accuracy = layer_info.get("final_test_accuracy", 0)
                        data[dataset_config_name][model][layer] = accuracy
        else:
            for config in configs:
                summary_path = os.path.join(dataset_path, config, "source_to_base", "all_layers_summary.json")
                if os.path.exists(summary_path):
                    dataset_config_name = f"{dataset} ({config})"
                    if dataset_config_name not in data:
                        data[dataset_config_name] = {}
                    data[dataset_config_name][model] = {}
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        for layer_info in summary.get("layers", []):
                            layer = layer_info.get("layer")
                            accuracy = layer_info.get("final_test_accuracy", 0)
                            data[dataset_config_name][model][layer] = accuracy

for dataset_config_name, models_data in data.items():
    df_data = {}
    for model, layers_data in models_data.items():
        target_layers = list(range(0, max_layer + 1, 5))
        full_layers = {l: np.nan for l in target_layers}
        for l in target_layers:
            if l in layers_data:
                full_layers[l] = layers_data[l]
        df_data[model] = full_layers
    
    df = pd.DataFrame(df_data)
    df.index.name = "Layer Index"
    
    df = df.reindex(sorted(df.columns), axis=1)
    

    df = df.sort_index(ascending=False)
    

    plt.figure(figsize=(10, 8))
    

    yticklabels = [str(l) for l in df.index]
    
    annot_df = df.map(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
    
    sns.heatmap(df, annot=annot_df, fmt="", cmap="magma_r", cbar_kws={'label': 'Final Test Accuracy'}, yticklabels=yticklabels)
    plt.title(dataset_config_name)
    plt.xlabel("Model Name")
    plt.ylabel("Layer Index")
    

    safe_filename = dataset_config_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{safe_filename}_heatmap.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Generated heatmap for {dataset_config_name} at {save_path}")
