import os, glob, re, json
import statistics

def get_data(model_name):
    data = {}
    
    paths = [
        # seed 42 (original)
        f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_es/{model_name}/obj_rel_across_anim_2/linear/4_attractor/base_to_source/layer_*/result.json",
        # seed 96
        f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_with_saved_boundaries/{model_name}/obj_rel_across_anim_2/linear/seed_96/4_attractor/base_to_source/layer_*/result.json",
        # seed 1120
        f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_with_saved_boundaries/{model_name}/obj_rel_across_anim_2/linear/seed_1120/4_attractor/base_to_source/layer_*/result.json"
    ]
    
    for pattern in paths:
        files = glob.glob(pattern)
        for f in files:
            m = re.search(r'layer_(\d+)', f)
            if m:
                layer = int(m.group(1))
                try:
                    with open(f, 'r') as fh:
                        res = json.load(fh)
                        val = res.get("training", {}).get("early_stopping", {}).get("best_test_accuracy")
                        if val is None:
                            # Try to find it elsewhere if not there
                            pass
                            
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

gemma_stats = get_data("gemma-3-12b-pt")
qwen_stats = get_data("qwen-3-8b")

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

