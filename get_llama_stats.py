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
            f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_with_saved_boundaries/{model_name}/{dataset}/{var}/4_attractor/base_to_source/layer_*/result.json",
            # seed 1120
            f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/bdas/results_das_full_with_saved_boundaries/{model_name}/{dataset}/{var}/4_attractor/base_to_source/layer_*/result.json"
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

d1 = get_data("llama-3.1-8b", "obj_rel_across_anim_2", "linear")
d2 = get_data("llama-3.1-8b", "obj_rel_across_anim_2", "bow")
d3 = get_data("llama-3.1-8b", "obj_rel_across_anim", "both")

for title, d in zip(["linear", "bow", "average"], [d1, d2, d3]):
    print(f"### {title}")
    for layer in sorted(d.keys()):
        print(f"| {layer} | {d[layer]['mean']:.4f} ± {d[layer]['std']:.4f} | count: {d[layer]['count']}")
    print()

