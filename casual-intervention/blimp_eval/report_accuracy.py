import argparse
import json
import os
from typing import Dict, List, Tuple


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_stats(records: List[Dict]) -> Dict[str, int | float]:
    total_pairs = 0
    correct_pairs = 0
    pred_a_count = 0
    pred_b_count = 0

    for record in records:
        pred = record.get("pair_pred")
        gold = record.get("pair_gold", "A")
        total_pairs += 1

        if pred == "A":
            pred_a_count += 1
        elif pred == "B":
            pred_b_count += 1

        if pred == gold:
            correct_pairs += 1

    wrong_pairs = total_pairs - correct_pairs
    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "wrong_pairs": wrong_pairs,
        "pred_a_count": pred_a_count,
        "pred_b_count": pred_b_count,
        "accuracy": accuracy,
    }


def collect_component_files(model_dir: str) -> List[Tuple[str, str]]:
    files = []
    for filename in os.listdir(model_dir):
        if filename.startswith("en_blimp_") and filename.endswith(".jsonl"):
            component = filename[len("en_blimp_") : -len(".jsonl")]
            files.append((component, os.path.join(model_dir, filename)))
    return sorted(files, key=lambda item: item[0])


def write_report(report_path: str, stats_by_component: Dict[str, Dict]) -> None:
    total_pairs = sum(stats["total_pairs"] for stats in stats_by_component.values())
    total_correct = sum(stats["correct_pairs"] for stats in stats_by_component.values())
    total_wrong = sum(stats["wrong_pairs"] for stats in stats_by_component.values())
    total_pred_a = sum(stats["pred_a_count"] for stats in stats_by_component.values())
    total_pred_b = sum(stats["pred_b_count"] for stats in stats_by_component.values())
    overall_accuracy = total_correct / total_pairs if total_pairs else 0.0

    lines = []
    lines.append("BLiMP pair-level accuracy report")
    lines.append("")
    lines.append("Overall:")
    lines.append(f"  total_pairs: {total_pairs}")
    lines.append(f"  correct_pairs: {total_correct}")
    lines.append(f"  wrong_pairs: {total_wrong}")
    lines.append(f"  total_predicted_A: {total_pred_a}")
    lines.append(f"  total_predicted_B: {total_pred_b}")
    lines.append(f"  accuracy: {overall_accuracy:.4f}")
    lines.append("")
    lines.append("Per component:")

    for component, stats in stats_by_component.items():
        lines.append(f"- {component}")
        lines.append(f"  total_pairs: {stats['total_pairs']}")
        lines.append(f"  correct_pairs: {stats['correct_pairs']}")
        lines.append(f"  wrong_pairs: {stats['wrong_pairs']}")
        lines.append(f"  predicted_A: {stats['pred_a_count']}")
        lines.append(f"  predicted_B: {stats['pred_b_count']}")
        lines.append(f"  accuracy: {stats['accuracy']:.4f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write BLiMP sentence-level accuracy report."
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b",
        help="Model id used for output directory naming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    base_dir= os.path.dirname(os.path.abspath(__file__))
    output_dir= os.path.join(base_dir, "output")
    report_dir= os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, sanitize_model_id(args.model))
    files = collect_component_files(model_dir)
    if not files:
        raise SystemExit(f"No BLiMP jsonl files found under {model_dir}")

    stats_by_component: Dict[str, Dict] = {}
    for component, path in files:
        records = load_jsonl(path)
        stats_by_component[component] = compute_stats(records)

    report_path = os.path.join(report_dir, f"blimp_accuracy_{model_name}.txt")
    write_report(report_path, stats_by_component)


if __name__ == "__main__":
    main()
