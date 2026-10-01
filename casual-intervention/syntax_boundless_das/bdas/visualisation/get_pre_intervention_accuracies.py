import json
import re
from pathlib import Path

# Paths
VIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VIS_DIR.parent.parent
RAW_RESULTS_DIR = (
    PROJECT_DIR / "data" / "data_generators" / "templates" / "raw_model_testing" / "results"
)
OUTPUT_FILE = VIS_DIR / "pre_intervention_accuracies.txt"

ATTRACTOR = 4

# Definitions
DIRECTIONS = {
    "source_to_base": {
        "setup": "SG -> PL",
        "baseline_metric": "base_overall_accuracy",
    },
    "base_to_source": {
        "setup": "PL -> SG",
        "baseline_metric": "source_overall_accuracy",
    },
}

HYPOTHESES = {
    "syn": {
        "name": "Syntax (H_syn)",
        "inputs": [("obj_rel_across_anim", "linear"), ("obj_rel_across_anim", "bow")],
    },
    "prox": {
        "name": "Proximity (H_prox)",
        "inputs": [("obj_rel_across_anim_2", "linear")],
    },
    "maj": {
        "name": "Majority (H_maj)",
        "inputs": [("obj_rel_across_anim_2", "bow")],
    },
}

MODELS = ["gemma-3-12b-pt", "qwen-3-8b"]


def raw_result_path(raw_results_dir, model, dataset, variation):
    return (
        raw_results_dir
        / model
        / f"{dataset}_test"
        / variation
        / f"{dataset}_pairs_{variation}_{ATTRACTOR}.txt"
    )


def read_raw_accuracy(path, metric_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing raw baseline file: {path}")

    pattern = re.compile(rf"^{re.escape(metric_name)}:\s*([0-9.]+)")
    with path.open("r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                return float(match.group(1))

    raise ValueError(f"Could not find {metric_name!r} in {path}")


def extract_pre_intervention_baseline(raw_results_dir, model, hypothesis_name, direction):
    metric_name = DIRECTIONS[direction]["baseline_metric"]
    accuracies = []
    for dataset, variation in HYPOTHESES[hypothesis_name]["inputs"]:
        path = raw_result_path(raw_results_dir, model, dataset, variation)
        accuracies.append(1.0 - read_raw_accuracy(path, metric_name))
    return sum(accuracies) / len(accuracies)


def main():
    lines = []
    lines.append("======================================================================")
    lines.append("                 Pre-Intervention Accuracy Baselines                  ")
    lines.append("======================================================================")
    lines.append("")

    for model in MODELS:
        lines.append(f"Model: {model}")
        lines.append("-" * len(f"Model: {model}"))
        
        for hyp_key, hyp_data in HYPOTHESES.items():
            lines.append(f"  Hypothesis: {hyp_data['name']}")
            
            for dir_key, dir_data in DIRECTIONS.items():
                try:
                    baseline = extract_pre_intervention_baseline(RAW_RESULTS_DIR, model, hyp_key, dir_key)
                    # Express as a percentage and decimal
                    percentage_str = f"{baseline * 100:.4f}%"
                    decimal_str = f"{baseline:.6f}"
                    lines.append(f"    Direction: {dir_data['setup']:<8} | Baseline Accuracy: {percentage_str:<10} ({decimal_str})")
                except Exception as e:
                    lines.append(f"    Direction: {dir_data['setup']:<8} | Error: {str(e)}")
            lines.append("")
        lines.append("=" * 70)
        lines.append("")

    content = "\n".join(lines)
    
    # Write to terminal
    print(content)
    
    # Save to file
    with OUTPUT_FILE.open("w") as f:
        f.write(content)
        
    print(f"Pre-intervention accuracy numbers successfully saved to:\n  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
