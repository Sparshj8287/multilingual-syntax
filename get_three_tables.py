import os, glob, re, json
import statistics

def get_data(model_name, dataset, variation):
    data = {}
    
    paths = []
    if variation == "both":
        # we will add both linear and bow
        variations = ["linear", "bow"]
    else:
        variations = [variation]
        
    for var in variations:
        paths.extend([
            # seed 42 (original)
            f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_es/{model_name}/{dataset}/{var}/4_attractor/base_to_source/layer_*/result.json",
            # seed 96
            f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_with_saved_boundaries/{model_name}/{dataset}/{var}/seed_96/4_attractor/base_to_source/layer_*/result.json",
            # seed 1120
            f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_with_saved_boundaries/{model_name}/{dataset}/{var}/seed_1120/4_attractor/base_to_source/layer_*/result.json"
        ])
        
    for pattern in paths:
        files = glob.glob(pattern)
        for f in files:
            m = re.search(r'layer_(\d+)', f)
            if m:
                layer = int(m.group(1))
                try:
                    with open(f, 'r') as fh:
                        res = json.load(fh)
                        val = res.get("best_test_accuracy")
                        if val is None:
                            val = res.get("training", {}).get("early_stopping", {}).get("best_test_accuracy")
                            
                        if val is not None:
                            if layer not in data:
                                data[layer] = []
                            data[layer].append(val)
                except Exception as e:
                    pass
    
    stats = {}
    for layer in data:
        if len(data[layer]) > 0:
            mean = statistics.mean(data[layer])
            std = statistics.stdev(data[layer]) if len(data[layer]) > 1 else 0.0
            stats[layer] = {
                "mean": mean,
                "std": std,
                "count": len(data[layer])
            }
    return stats

def print_table(title, gemma_stats, qwen_stats):
    print(f"### {title}")
    all_layers = sorted(list(set(gemma_stats.keys()) | set(qwen_stats.keys())))
    
    print("| Layer | Gemma 3 12B PT | Qwen 3 8B |")
    print("|---|---|---|")
    for layer in all_layers:
        if layer in gemma_stats:
            g_val = f"{gemma_stats[layer]['mean']:.4f} ± {gemma_stats[layer]['std']:.4f}"
        else:
            g_val = "N/A"
            
        if layer in qwen_stats:
            q_val = f"{qwen_stats[layer]['mean']:.4f} ± {qwen_stats[layer]['std']:.4f}"
        else:
            q_val = "N/A"
            
        print(f"| {layer} | {g_val} | {q_val} |")
    print()

# Table 1: obj_rel_across_anim_2 / linear
print_table(
    "obj_rel_across_anim_2 / linear",
    get_data("gemma-3-12b-pt", "obj_rel_across_anim_2", "linear"),
    get_data("qwen-3-8b", "obj_rel_across_anim_2", "linear")
)

# Table 2: obj_rel_across_anim_2 / bow
print_table(
    "obj_rel_across_anim_2 / bow",
    get_data("gemma-3-12b-pt", "obj_rel_across_anim_2", "bow"),
    get_data("qwen-3-8b", "obj_rel_across_anim_2", "bow")
)

# Table 3: obj_rel_across_anim / average of linear and bow
print_table(
    "obj_rel_across_anim (average of linear and bow)",
    get_data("gemma-3-12b-pt", "obj_rel_across_anim", "both"),
    get_data("qwen-3-8b", "obj_rel_across_anim", "both")
)

