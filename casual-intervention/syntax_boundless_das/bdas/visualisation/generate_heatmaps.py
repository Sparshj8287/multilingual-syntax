import argparse
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

results_dir = Path(__file__).resolve().parent.parent / "results_das_full_es"
output_dir = Path(__file__).resolve().parent / "plots_das_full_es"

output_dir.mkdir(exist_ok=True)

MODEL_ALIASES = {
    "gemma": "gemma",
    "qwen": "qwen",
    "olmo": "olmo",
    "llama": "llama",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DAS heatmaps from all_layers_summary.json files."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model aliases/names to include, e.g. --models gemma qwen. "
            "Use --models all to include every model."
        ),
    )
    return parser.parse_args()


def normalize_model_filters(raw_models):
    models = []
    for raw_model in raw_models:
        models.extend(part.strip().lower() for part in raw_model.split(","))
    models = [model for model in models if model]
    if not models or "all" in models:
        return None

    invalid_models = sorted(
        set(model for model in models if model not in MODEL_ALIASES)
    )
    if invalid_models:
        valid_names = ", ".join(["all", *MODEL_ALIASES])
        raise ValueError(
            f"Unknown model filter(s): {', '.join(invalid_models)}. "
            f"Use one or more of: {valid_names}."
        )
    return tuple(MODEL_ALIASES[model] for model in models)


def should_include_model(model, model_filters):
    if model_filters is None:
        return True
    normalized_model = model.lower()
    return any(model_filter in normalized_model for model_filter in model_filters)


def build_group_key(paradigm, hypothesis, direction, config_name=None):
    parts = [paradigm, hypothesis]
    if config_name:
        parts.append(config_name)
    parts.append(direction)
    filename_stem = "_".join(parts)
    title_parts = [
        f"Paradigm: {paradigm}",
        f"Hypothesis: {hypothesis}",
        f"Direction: {direction}",
    ]
    if config_name:
        title_parts.append(f"Config: {config_name}")
    return filename_stem, " | ".join(title_parts)


# data[group_key][column_name][layer] = accuracy
args = parse_args()
model_filters = normalize_model_filters(args.models)
if model_filters is None:
    print("[models] including all models")
else:
    print(f"[models] including: {', '.join(model_filters)}")

data = {}
group_titles = {}

# We recursively find all all_layers_summary.json files
for summary_path in results_dir.rglob("all_layers_summary.json"):
    # Expected path parts: results / {model} / {dataset} / {variation} / {nua}_attractor / {direction} / all_layers_summary.json
    parts = summary_path.relative_to(results_dir).parts
    if len(parts) < 6:
        continue
        
    model = parts[0]
    if not should_include_model(model, model_filters):
        continue

    dataset = parts[1]
    variation = parts[2]
    nua_dir = parts[3]
    
    # Check if there is an extra config folder before direction
    if parts[4] not in ["source_to_base", "base_to_source"]:
        if len(parts) >= 7 and parts[5] in ["source_to_base", "base_to_source"]:
            config_name = parts[4]
            direction = parts[5]
        else:
            continue
    else:
        config_name = None
        direction = parts[4]

    group_key, group_title = build_group_key(
        dataset,
        variation,
        direction,
        config_name=config_name,
    )
    
    col_name = f"{model} ({nua_dir})"
    
    if group_key not in data:
        data[group_key] = {}
        group_titles[group_key] = group_title
    if col_name not in data[group_key]:
        data[group_key][col_name] = {}
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        for layer_info in summary.get("layers", []):
            layer = layer_info.get("layer")
            accuracy = layer_info.get("final_test_accuracy", 0)
            data[group_key][col_name][layer] = accuracy

if not data:
    print(f"No summary files found in {results_dir}")
    exit(0)

for group_key, cols_data in data.items():
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
    plt.title(group_titles.get(group_key, group_key))
    plt.xlabel("Model and Attractors")
    plt.ylabel("Layer Index")
    
    safe_filename = group_key.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"{safe_filename}_heatmap.png"
    save_path = output_dir / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Generated heatmap for {group_titles.get(group_key, group_key)} at {save_path}")
