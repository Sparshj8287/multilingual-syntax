import os, glob, re
import statistics

def get_data(base_paths, model_name):
    data = {}
    for base_path in base_paths:
        pattern = os.path.join(base_path, model_name, "layer-*", "parse-distance", "*", "test.uuas")
        files = glob.glob(pattern)
        for f in files:
            # Extract layer number
            m = re.search(r'layer-(\d+)', f)
            if m:
                layer = int(m.group(1))
                with open(f, 'r') as fh:
                    try:
                        val = float(fh.read().strip())
                        if layer not in data:
                            data[layer] = []
                        data[layer].append(val)
                    except:
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

base_paths = [
    "/home/sparsh/projects/multilingual-syntax/multilingual-probing-visualization/experiments/syntax_boundless/obj_rel_across_anim/bow/base_sentence/attractors_4",
    "/home/sparsh/projects/multilingual-syntax/multilingual-probing-visualization/experiments/syntax_boundless/seed_96/obj_rel_across_anim/bow/base_sentence/attractors_4",
    "/home/sparsh/projects/multilingual-syntax/multilingual-probing-visualization/experiments/syntax_boundless/seed_1120/obj_rel_across_anim/bow/base_sentence/attractors_4"
]

gemma_stats = get_data(base_paths, "google-gemma-3-12b-pt")
qwen_stats = get_data(base_paths, "Qwen-Qwen3-8B-Base")

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

