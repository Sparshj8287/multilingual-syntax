import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-directory model accuracies from result text files."
    )
    parser.add_argument(
        "--results-dir",
        default=(
            "/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results"
        ),
        help="Directory containing model result folders.",
    )
    parser.add_argument(
        "--plots-dir",
        default=(
            "/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results/plots"
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
    return path.replace(os.sep, "__").replace(".", "_")


def collect_results_by_directory(
    results_dir: str,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    # directory_key -> file_key -> model_name -> (base_acc, source_acc)
    results: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    if not os.path.isdir(results_dir):
        return results

    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir) or model_name == "plots":
            continue
        for root, _, files in os.walk(model_dir):
            for name in files:
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(root, name)
                rel_path = os.path.relpath(path, model_dir)
                rel_dir = os.path.dirname(rel_path)
                directory_key = rel_dir if rel_dir not in ("", ".") else "root"
                file_key = os.path.splitext(os.path.basename(rel_path))[0]
                parsed = parse_result_file(path)
                if parsed is None:
                    continue
                results.setdefault(directory_key, {}).setdefault(file_key, {})[
                    model_name
                ] = parsed
    return results


def _build_matrix(
    directory_results: dict[str, dict[str, tuple[float, float]]],
    files: list[str],
    models: list[str],
    idx: int,
) -> np.ndarray:
    matrix = np.full((len(files), len(models)), np.nan, dtype=float)
    for i, file_key in enumerate(files):
        for j, model_name in enumerate(models):
            if model_name in directory_results[file_key]:
                matrix[i, j] = directory_results[file_key][model_name][idx]
    return matrix


def plot_directory_summary(
    directory_key: str,
    directory_results: dict[str, dict[str, tuple[float, float]]],
    plots_dir: str,
) -> None:
    files = sorted(directory_results.keys())
    models = sorted(
        {model_name for file_key in files for model_name in directory_results[file_key]}
    )
    if not files or not models:
        return

    base_mat = _build_matrix(directory_results, files, models, idx=0)
    source_mat = _build_matrix(directory_results, files, models, idx=1)

    fig_h = max(5, len(files) * 0.55)
    fig_w = max(10, len(models) * 1.25)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    im0 = axes[0].imshow(base_mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    im1 = axes[1].imshow(source_mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")

    axes[0].set_title("Base Overall Accuracy")
    axes[1].set_title("Source Overall Accuracy")

    for ax in axes:
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(files)))
        ax.set_yticklabels(files)
        ax.set_xlabel("Model")
    axes[0].set_ylabel("Result File")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Directory Summary: {directory_key}", fontsize=12)

    os.makedirs(plots_dir, exist_ok=True)
    out_name = f"{sanitize_filename(directory_key)}_summary.png"
    out_path = os.path.join(plots_dir, out_name)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grouped_results = collect_results_by_directory(args.results_dir)
    if not grouped_results:
        raise SystemExit(f"No result files found in {args.results_dir}")
    for directory_key, directory_results in grouped_results.items():
        plot_directory_summary(directory_key, directory_results, args.plots_dir)
    print(f"Saved {len(grouped_results)} directory-level plot(s) to: {args.plots_dir}")


if __name__ == "__main__":
    main()
