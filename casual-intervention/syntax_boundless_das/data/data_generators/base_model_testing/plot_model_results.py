import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model accuracies from batch evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        default=(
            "casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results"
        ),
        help="Directory containing model result folders.",
    )
    parser.add_argument(
        "--plots-dir",
        default=(
            "casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results/plots"
        ),
        help="Directory to save plots.",
    )
    return parser.parse_args()


def parse_result_file(path: str) -> tuple[float, float] | None:
    base_acc = None
    source_acc = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("base_overall_accuracy:"):
                base_acc = float(line.split(":", 1)[1].strip())
            elif line.startswith("source_overall_accuracy:"):
                source_acc = float(line.split(":", 1)[1].strip())
            if base_acc is not None and source_acc is not None:
                break
    if base_acc is None or source_acc is None:
        return None
    return base_acc, source_acc


def sanitize_filename(path: str) -> str:
    return path.replace(os.sep, "__")


def collect_results(results_dir: str) -> dict:
    results: dict[str, dict[str, tuple[float, float]]] = {}
    if not os.path.isdir(results_dir):
        return results

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for root, _, files in os.walk(model_dir):
            for name in files:
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, model_dir)
                jsonl_key = os.path.splitext(rel_path)[0]
                parsed = parse_result_file(path)
                if parsed is None:
                    continue
                results.setdefault(jsonl_key, {})[model_name] = parsed
    return results


def plot_jsonl_results(
    jsonl_key: str,
    model_results: dict[str, tuple[float, float]],
    plots_dir: str,
) -> None:
    models = sorted(model_results.keys())
    if not models:
        return

    base_acc = [model_results[m][0] for m in models]
    source_acc = [model_results[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(max(8, len(models) * 1.6), 5))
    plt.bar(x - width / 2, base_acc, width, label="Base accuracy")
    plt.bar(x + width / 2, source_acc, width, label="Source accuracy")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(jsonl_key)
    plt.xticks(x, models, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    out_name = sanitize_filename(jsonl_key) + "_accuracy.png"
    out_path = os.path.join(plots_dir, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    results = collect_results(args.results_dir)
    if not results:
        raise SystemExit(f"No result files found in {args.results_dir}")
    for jsonl_key, model_results in results.items():
        plot_jsonl_results(jsonl_key, model_results, args.plots_dir)


if __name__ == "__main__":
    main()
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model accuracies from batch evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        default=(
            "casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results"
        ),
        help="Directory containing model result folders.",
    )
    parser.add_argument(
        "--plots-dir",
        default=(
            "casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results/plots"
        ),
        help="Directory to save plots.",
    )
    return parser.parse_args()


def parse_result_file(path: str) -> tuple[float, float] | None:
    base_acc = None
    source_acc = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("base_overall_accuracy:"):
                base_acc = float(line.split(":", 1)[1].strip())
            elif line.startswith("source_overall_accuracy:"):
                source_acc = float(line.split(":", 1)[1].strip())
            if base_acc is not None and source_acc is not None:
                break
    if base_acc is None or source_acc is None:
        return None
    return base_acc, source_acc


def sanitize_filename(path: str) -> str:
    return path.replace(os.sep, "__")


def collect_results(results_dir: str) -> dict:
    results: dict[str, dict[str, tuple[float, float]]] = {}
    if not os.path.isdir(results_dir):
        return results

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for root, _, files in os.walk(model_dir):
            for name in files:
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, model_dir)
                jsonl_key = os.path.splitext(rel_path)[0]
                parsed = parse_result_file(path)
                if parsed is None:
                    continue
                results.setdefault(jsonl_key, {})[model_name] = parsed
    return results


def plot_jsonl_results(
    jsonl_key: str,
    model_results: dict[str, tuple[float, float]],
    plots_dir: str,
) -> None:
    models = sorted(model_results.keys())
    if not models:
        return

    base_acc = [model_results[m][0] for m in models]
    source_acc = [model_results[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(max(8, len(models) * 1.6), 5))
    plt.bar(x - width / 2, base_acc, width, label="Base accuracy")
    plt.bar(x + width / 2, source_acc, width, label="Source accuracy")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(jsonl_key)
    plt.xticks(x, models, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    out_name = sanitize_filename(jsonl_key) + "_accuracy.png"
    out_path = os.path.join(plots_dir, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    results = collect_results(args.results_dir)
    if not results:
        raise SystemExit(f"No result files found in {args.results_dir}")
    for jsonl_key, model_results in results.items():
        plot_jsonl_results(jsonl_key, model_results, args.plots_dir)


if __name__ == "__main__":
    main()
