import argparse
import json
import math
import re
from pathlib import Path

try:
    from matplotlib import font_manager
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    raise SystemExit(1)


VIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VIS_DIR.parent.parent
DEFAULT_RESULTS_DIR = VIS_DIR.parent / "corrupted_results_das_full_with_saved_boundaries"
DEFAULT_RAW_RESULTS_DIR = (
    PROJECT_DIR / "data" / "data_generators" / "templates" / "raw_model_testing" / "results"
)
DEFAULT_PLOTS_DIR = VIS_DIR / "plots_corrupted_lines"

ATTRACTOR = 4
DATASET = "corrupted_obj_rel_across_anim"
VARIATION = "linear"
# The corrupted control keeps every base sentence and reuses the same source
# sentence pool, so the pre-intervention baselines of the uncorrupted dataset
# still apply.
BASELINE_DATASET = "obj_rel_across_anim"
DIRECTIONS = {
    "source_to_base": {
        "setup": "SG $\\rightarrow$ PL",
        "baseline_metric": "base_overall_accuracy",
    },
    "base_to_source": {
        "setup": "PL $\\rightarrow$ SG",
        "baseline_metric": "source_overall_accuracy",
    },
}

SERIES_LABEL = r"$\mathcal{H}_{ctrl}$ (Scrambled Target)"
SERIES_COLOR = "#7E6148"
SERIES_MARKER = "D"
MODEL_ALIASES = {
    "gemma": "gemma",
    "qwen": "qwen",
    "olmo": "olmo",
    "llama": "llama",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Boundless DAS line plots for the scrambled-target control "
            "run at attractor 4. SG -> PL is read from source_to_base, and "
            "PL -> SG from base_to_source."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Corrupted DAS results directory. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--raw-results-dir",
        type=Path,
        default=DEFAULT_RAW_RESULTS_DIR,
        help=(
            "Raw model testing results directory used for pre-intervention baselines. "
            f"Default: {DEFAULT_RAW_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Output directory for line plots. Default: {DEFAULT_PLOTS_DIR}",
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
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Plot formats to write, e.g. --formats png pdf.",
    )
    return parser.parse_args()


def normalize_model_filters(raw_models):
    models = []
    for raw_model in raw_models:
        models.extend(part.strip().lower() for part in raw_model.split(","))
    models = [model for model in models if model]
    if not models or "all" in models:
        return None

    return tuple(MODEL_ALIASES.get(model, model) for model in models)


def should_include_model(model, model_filters):
    if model_filters is None:
        return True
    normalized_model = model.lower()
    return any(model_filter in normalized_model for model_filter in model_filters)


def format_model_name(model):
    parts = [part for part in model.replace("_", "-").split("-") if part and part != "pt"]
    formatted_parts = []
    for part in parts:
        if part.isdigit():
            formatted_parts.append(part)
        elif re.fullmatch(r"\d+b", part, flags=re.IGNORECASE):
            formatted_parts.append(part[:-1] + "B")
        else:
            formatted_parts.append(part.capitalize())
    return "-".join(formatted_parts)


def model_sort_key(model):
    order = {"gemma": 0, "qwen": 1, "olmo": 2, "llama": 3}
    family = model.split("-", 1)[0].lower()
    return (order.get(family, 99), model)


def configure_plot_style():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ]
    font_family = next(
        (font for font in preferred_fonts if font in available_fonts),
        "DejaVu Serif",
    )

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20,
        }
    )


def test_set_size(summary_path, layer_info):
    result_file = layer_info.get("result_file")
    candidates = []
    if result_file:
        candidates.append(Path(result_file))
    layer = layer_info.get("layer")
    if layer is not None:
        candidates.append(summary_path.parent / f"layer_{layer}" / "result.json")

    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r") as f:
            result = json.load(f)
        total = result.get("final", {}).get("test", {}).get("total")
        if total:
            return int(total)
    return None


def binomial_standard_error(accuracy, sample_size):
    if not sample_size:
        return 0.0
    return math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / sample_size)


def parse_results_tree(results_dir, model_filters):
    data = {}
    for summary_path in results_dir.rglob("all_layers_summary.json"):
        parts = summary_path.relative_to(results_dir).parts
        if len(parts) < 6:
            continue

        model, dataset, variation, attractor_dir, direction = parts[:5]
        if dataset != DATASET or variation != VARIATION:
            continue
        if direction not in DIRECTIONS:
            continue
        if not should_include_model(model, model_filters):
            continue

        try:
            attractor = int(attractor_dir.split("_", 1)[0])
        except ValueError:
            continue
        if attractor != ATTRACTOR:
            continue

        with summary_path.open("r") as f:
            summary = json.load(f)

        series = {}
        standard_errors = {}
        for layer_info in summary.get("layers", []):
            layer = layer_info.get("layer")
            accuracy = layer_info.get("final_test_accuracy")
            if layer is None or accuracy is None:
                continue
            layer = int(layer)
            accuracy = float(accuracy)
            series[layer] = accuracy
            standard_errors[layer] = binomial_standard_error(
                accuracy, test_set_size(summary_path, layer_info)
            )

        if series:
            data.setdefault(model, {})[direction] = {
                "series": series,
                "std": standard_errors,
            }

    return data


def raw_result_path(raw_results_dir, model):
    return (
        raw_results_dir
        / model
        / f"{BASELINE_DATASET}_test"
        / VARIATION
        / f"{BASELINE_DATASET}_pairs_{VARIATION}_{ATTRACTOR}.txt"
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


def build_plot_data(data, raw_results_dir):
    plot_data = {}
    for model, model_data in data.items():
        model_plot_data = {}
        for direction, direction_data in model_data.items():
            metric_name = DIRECTIONS[direction]["baseline_metric"]
            try:
                baseline = 1.0 - read_raw_accuracy(
                    raw_result_path(raw_results_dir, model), metric_name
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"[baseline missing] {model} {direction}: {error}")
                baseline = None

            model_plot_data[direction] = {**direction_data, "baseline": baseline}

        if model_plot_data:
            plot_data[model] = model_plot_data

    return plot_data


def plot_combined_models(plot_data, plots_dir, formats):
    panels = [
        (model, direction)
        for model in sorted(plot_data, key=model_sort_key)
        for direction in DIRECTIONS
        if direction in plot_data[model]
    ]
    if not panels:
        return []

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5 * len(panels), 4.8),
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    show_baseline_legend = False
    for idx, (model, direction) in enumerate(panels):
        panel_data = plot_data[model][direction]
        ax = axes[idx]
        layers = sorted(panel_data["series"])
        values = [panel_data["series"][layer] * 100 for layer in layers]
        standard_errors = [panel_data["std"].get(layer, 0.0) * 100 for layer in layers]

        ax.plot(
            layers,
            values,
            marker=SERIES_MARKER,
            color=SERIES_COLOR,
            linewidth=2.5,
            markersize=9,
        )
        if any(standard_errors):
            ax.fill_between(
                layers,
                [value - std for value, std in zip(values, standard_errors)],
                [value + std for value, std in zip(values, standard_errors)],
                color=SERIES_COLOR,
                alpha=0.2,
                linewidth=0,
            )
        if panel_data["baseline"] is not None:
            show_baseline_legend = True
            ax.axhline(
                panel_data["baseline"] * 100,
                color=SERIES_COLOR,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
            )

        ax.set_ylim(0, 105)
        if layers:
            ax.set_xlim(min(layers) - 2, max(layers) + 2)
        ax.set_xticks(
            [layer for layer in [0, 10, 20, 30, 40] if not layers or layer <= max(layers)]
        )
        panel_label = (
            f"({chr(97 + idx)}) {format_model_name(model)} "
            f"({DIRECTIONS[direction]['setup']})"
        )
        ax.set_xlabel(f"Layer Index\n\n{panel_label}")
        ax.yaxis.grid(True, linestyle="-", alpha=0.2)

        if idx == 0:
            ax.set_ylabel("IIA (%)")
            ax.set_yticks([0, 25, 50, 75, 100])

    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color=SERIES_COLOR,
            marker=SERIES_MARKER,
            linewidth=2.5,
            markersize=9,
            label=SERIES_LABEL,
        )
    ]
    if show_baseline_legend:
        legend_elements.append(
            mlines.Line2D(
                [0],
                [0],
                color="#808080",
                linestyle="--",
                linewidth=1.5,
                label="Pre-Intervention Accuracy",
            )
        )
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.subplots_adjust(wspace=0.08, bottom=0.42)

    saved_paths = []
    model_suffix = "_".join(model.split("-", 1)[0].lower() for model, _direction in panels)
    model_suffix = "_".join(dict.fromkeys(model_suffix.split("_")))
    for plot_format in formats:
        suffix = plot_format.lower().lstrip(".")
        save_path = (
            plots_dir
            / f"{model_suffix}_attractor_{ATTRACTOR}_corrupted_das_lineplot.{suffix}"
        )
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        saved_paths.append(save_path)

    plt.close(fig)
    return saved_paths


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    raw_results_dir = args.raw_results_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_filters = normalize_model_filters(args.models)
    if model_filters is None:
        print("[models] including all models")
    else:
        print(f"[models] including: {', '.join(model_filters)}")

    data = parse_results_tree(results_dir, model_filters)
    if not data:
        print(f"No attractor {ATTRACTOR} all_layers_summary.json files found in {results_dir}")
        return

    configure_plot_style()
    plot_data = build_plot_data(data, raw_results_dir)
    if not plot_data:
        print("No complete model/setup combinations found to plot.")
        return

    saved_paths = plot_combined_models(plot_data, plots_dir, args.formats)
    for save_path in saved_paths:
        print(f"Generated line plot at {save_path}")


if __name__ == "__main__":
    main()
