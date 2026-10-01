import argparse
import json
import math
import re
import statistics
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
DEFAULT_SEED_42_RESULTS_DIR = VIS_DIR.parent / "results_das_full"
DEFAULT_SEED_42_FALLBACK_RESULTS_DIR = VIS_DIR.parent / "results_das_full_es"
DEFAULT_RAW_RESULTS_DIR = (
    PROJECT_DIR / "data" / "data_generators" / "templates" / "raw_model_testing" / "results"
)
DEFAULT_PLOTS_DIR = VIS_DIR / "plots_std_lines"

ATTRACTOR = 4
BASELINE_SEED = "seed_42"
EXPECTED_SEEDS = (BASELINE_SEED, "seed_96", "seed_1120")
DIRECTIONS = {
    "source_to_base": {
        "setup": "SG $\\rightarrow$ PL",
        "filename": "sg_to_pl",
        "baseline_metric": "base_overall_accuracy",
    },
    "base_to_source": {
        "setup": "PL $\\rightarrow$ SG",
        "filename": "pl_to_sg",
        "baseline_metric": "source_overall_accuracy",
    },
}
HYPOTHESES = {
    "syn": {
        "label": r"$\mathcal{H}_{syn}$ (Syntax)",
        "inputs": [("obj_rel_across_anim", "linear"), ("obj_rel_across_anim", "bow")],
    },
    "prox": {
        "label": r"$\mathcal{H}_{prox}$ (Proximity)",
        "inputs": [("obj_rel_across_anim_2", "linear")],
    },
    "maj": {
        "label": r"$\mathcal{H}_{maj}$ (Majority)",
        "inputs": [("obj_rel_across_anim_2", "bow")],
    },
}

# User-requested colors: syntax green, proximity red, majority blue.
COLORS = {
    "syn": "#00A087",
    "prox": "#E64B35",
    "maj": "#3C5488",
}
MARKERS = {"syn": "o", "prox": "s", "maj": "^"}
MODEL_ALIASES = {
    "gemma": "gemma",
    "qwen": "qwen",
    "olmo": "olmo",
    "llama": "llama",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Boundless DAS line plots for attractor 4. "
            "SG -> PL is read from source_to_base, and PL -> SG from base_to_source."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"DAS results directory containing model/dataset/variation folders. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--seed-42-results-dir",
        type=Path,
        default=DEFAULT_SEED_42_RESULTS_DIR,
        help=(
            "Complete unprefixed seed-42 DAS results directory. Seeds 96 and 1120 "
            f"are read from --results-dir. Default: {DEFAULT_SEED_42_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--seed-42-fallback-results-dir",
        type=Path,
        default=DEFAULT_SEED_42_FALLBACK_RESULTS_DIR,
        help=(
            "Fallback seed-42 directory for combinations absent from "
            f"--seed-42-results-dir. Default: {DEFAULT_SEED_42_FALLBACK_RESULTS_DIR}"
        ),
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


def parse_results_tree(
    results_dir,
    seed_42_results_dir,
    seed_42_fallback_results_dir,
    model_filters,
):
    data = {}

    # Parse the fallback first so primary seed-42 results overwrite it.
    result_sources = (
        (seed_42_fallback_results_dir, BASELINE_SEED),
        (seed_42_results_dir, BASELINE_SEED),
        (results_dir, None),
    )
    for source_dir, forced_seed in result_sources:
        for summary_path in source_dir.rglob("all_layers_summary.json"):
            parts = summary_path.relative_to(source_dir).parts
            if len(parts) < 6:
                continue

            model = parts[0]
            dataset = parts[1]
            variation = parts[2]
            path_seed = parts[3] if parts[3].startswith("seed_") else BASELINE_SEED
            if forced_seed is not None and path_seed != forced_seed:
                continue
            if forced_seed is None and path_seed == BASELINE_SEED:
                continue
            seed_name = forced_seed or path_seed
            attractor_idx = 4 if seed_name != BASELINE_SEED else 3
            if len(parts) <= attractor_idx + 2:
                continue
            attractor_dir = parts[attractor_idx]
            direction_idx = attractor_idx + 1

            if not should_include_model(model, model_filters):
                continue

            try:
                attractor = int(attractor_dir.split("_", 1)[0])
            except ValueError:
                continue

            if attractor != ATTRACTOR:
                continue

            if parts[direction_idx] not in DIRECTIONS:
                direction_idx += 1
                if len(parts) <= direction_idx or parts[direction_idx] not in DIRECTIONS:
                    continue

            direction = parts[direction_idx]
            data.setdefault(model, {}).setdefault(direction, {}).setdefault(
                dataset, {}
            ).setdefault(variation, {}).setdefault(seed_name, {})

            with summary_path.open("r") as f:
                summary = json.load(f)
            for layer_info in summary.get("layers", []):
                layer = layer_info.get("layer")
                accuracy = layer_info.get("final_test_accuracy")
                if layer is None or accuracy is None:
                    continue
                data[model][direction][dataset][variation][seed_name][int(layer)] = float(
                    accuracy
                )

    return data


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


def average_layer_series(series_list):
    layers = sorted(set().union(*(series.keys() for series in series_list)))
    averaged = {}
    for layer in layers:
        values = [series[layer] for series in series_list if layer in series]
        averaged[layer] = sum(values) / len(values)
    return averaged


def extract_hypothesis_seed_series(model_data, hypothesis_name, direction):
    input_seed_series = []
    for dataset, variation in HYPOTHESES[hypothesis_name]["inputs"]:
        seed_series = model_data.get(direction, {}).get(dataset, {}).get(variation)
        if not seed_series:
            raise KeyError(
                f"Missing DAS data for {direction}/{dataset}/{variation}/{ATTRACTOR}_attractor"
            )
        input_seed_series.append(seed_series)

    missing_seeds = [
        seed
        for seed in EXPECTED_SEEDS
        if any(seed not in seed_series for seed_series in input_seed_series)
    ]
    if missing_seeds:
        raise KeyError(f"Missing DAS results for seeds: {', '.join(missing_seeds)}")

    return {
        seed: average_layer_series(
            [seed_series[seed] for seed_series in input_seed_series]
        )
        for seed in EXPECTED_SEEDS
    }


def aggregate_seed_series(seed_series):
    common_layers = sorted(
        set.intersection(*(set(series) for series in seed_series.values()))
    )
    baseline = {}
    standard_deviations = {}
    for layer in common_layers:
        values = [seed_series[seed][layer] for seed in EXPECTED_SEEDS]
        baseline[layer] = seed_series[BASELINE_SEED][layer]
        standard_deviations[layer] = statistics.stdev(values)
    return baseline, standard_deviations


def extract_pre_intervention_baseline(raw_results_dir, model, hypothesis_name, direction):
    metric_name = DIRECTIONS[direction]["baseline_metric"]
    accuracies = []
    for dataset, variation in HYPOTHESES[hypothesis_name]["inputs"]:
        path = raw_result_path(raw_results_dir, model, dataset, variation)
        accuracies.append(1.0 - read_raw_accuracy(path, metric_name))
    return sum(accuracies) / len(accuracies)


def build_plot_data(data, raw_results_dir):
    plot_data = {}
    for model, model_data in data.items():
        model_plot_data = {}
        for direction in DIRECTIONS:
            direction_plot_data = {}
            for hypothesis in HYPOTHESES:
                try:
                    seed_series = extract_hypothesis_seed_series(
                        model_data, hypothesis, direction
                    )
                    series, standard_deviations = aggregate_seed_series(seed_series)
                    baseline = extract_pre_intervention_baseline(
                        raw_results_dir, model, hypothesis, direction
                    )
                except (FileNotFoundError, KeyError, ValueError) as error:
                    print(f"[skip] {model} {direction} {hypothesis}: {error}")
                    direction_plot_data = {}
                    break

                direction_plot_data[hypothesis] = {
                    "series": series,
                    "std": standard_deviations,
                    "baseline": baseline,
                }

            if direction_plot_data:
                model_plot_data[direction] = direction_plot_data

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

    for idx, (model, direction) in enumerate(panels):
        model_plot_data = plot_data[model]
        direction_config = DIRECTIONS[direction]
        ax = axes[idx]
        layers = sorted(
            set().union(
                *[
                    hypothesis_data["series"].keys()
                    for hypothesis_data in model_plot_data[direction].values()
                ]
            )
        )

        for hypothesis, config in HYPOTHESES.items():
            hypothesis_data = model_plot_data[direction][hypothesis]
            series = hypothesis_data["series"]
            values = [
                series[layer] * 100 if layer in series else math.nan
                for layer in layers
            ]
            standard_deviations = [
                hypothesis_data["std"].get(layer, 0.0) * 100 for layer in layers
            ]

            ax.plot(
                layers,
                values,
                marker=MARKERS[hypothesis],
                color=COLORS[hypothesis],
                linewidth=2.5,
                markersize=9,
            )
            if any(standard_deviations):
                ax.fill_between(
                    layers,
                    [value - std for value, std in zip(values, standard_deviations)],
                    [value + std for value, std in zip(values, standard_deviations)],
                    color=COLORS[hypothesis],
                    alpha=0.2,
                    linewidth=0,
                )
            ax.axhline(
                hypothesis_data["baseline"] * 100,
                color=COLORS[hypothesis],
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
        panel_label = f"({chr(97 + idx)}) {format_model_name(model)} ({direction_config['setup']})"
        ax.set_xlabel(f"Layer Index\n\n{panel_label}")
        ax.yaxis.grid(True, linestyle="-", alpha=0.2)

        if idx == 0:
            ax.set_ylabel("IIA (%)")
            ax.set_yticks([0, 25, 50, 75, 100])

    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color=COLORS[hypothesis],
            marker=MARKERS[hypothesis],
            linewidth=2.5,
            markersize=9,
            label=config["label"],
        )
        for hypothesis, config in HYPOTHESES.items()
    ]
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
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),

    )
    plt.subplots_adjust(wspace=0.08, bottom=0.42)

    saved_paths = []
    model_suffix = "_".join(model.split("-", 1)[0].lower() for model, _direction in panels)
    # Keep the filename compact when each model contributes both setups.
    model_suffix = "_".join(dict.fromkeys(model_suffix.split("_")))
    for plot_format in formats:
        suffix = plot_format.lower().lstrip(".")
        save_path = plots_dir / f"{model_suffix}_attractor_{ATTRACTOR}_all_setups_das_lineplot.{suffix}"
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        saved_paths.append(save_path)

    plt.close(fig)
    return saved_paths


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    seed_42_results_dir = args.seed_42_results_dir.expanduser().resolve()
    seed_42_fallback_results_dir = (
        args.seed_42_fallback_results_dir.expanduser().resolve()
    )
    raw_results_dir = args.raw_results_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_filters = normalize_model_filters(args.models)
    if model_filters is None:
        print("[models] including all models")
    else:
        print(f"[models] including: {', '.join(model_filters)}")

    data = parse_results_tree(
        results_dir,
        seed_42_results_dir,
        seed_42_fallback_results_dir,
        model_filters,
    )
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
