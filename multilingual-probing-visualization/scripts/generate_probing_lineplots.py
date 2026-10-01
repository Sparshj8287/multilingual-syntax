import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

try:
    from matplotlib import font_manager
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    raise SystemExit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
EXPERIMENT_PRESETS = {
    "model-results": (
        EXPERIMENTS_DIR / "gemma-3-12b-pt",
        EXPERIMENTS_DIR / "llama-3.1-8b",
        EXPERIMENTS_DIR / "qwen-3-8b",
    ),
    "syntax-boundless": (EXPERIMENTS_DIR / "syntax_boundless",),
}
DEFAULT_PLOTS_DIR = PROJECT_DIR / "visualizations_plots" / "probing_lineplots"

ATTRACTOR_RE = re.compile(r"^attractors_(\d+)$")
LAYER_RE = re.compile(r"(?:^|-)layer-(\d+)$")
FLAT_RESULT_RE = re.compile(r"^model-layer-(\d+)-(.+)-\d{4}-\d+-\d+-.*$")
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


@dataclass(frozen=True)
class MetricPoint:
    dataset_name: str
    variation: str
    sentence_type: str
    attractor_count: int
    seed_name: str
    model_name: str
    layer: int
    task_name: str
    metric_path: Path
    modified_time: float
    layout: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate probing line plots from test.uuas files. "
            "Syntax is the layer-wise average of obj_rel_across_anim linear/bow; "
            "proximity and majority use obj_rel_across_anim_2 linear/bow."
        )
    )
    parser.add_argument(
        "--experiment-preset",
        choices=tuple(EXPERIMENT_PRESETS),
        default="model-results",
        help=(
            "Named input layout. 'model-results' uses the Gemma, Llama, and Qwen "
            "result directories; 'syntax-boundless' restores the prior layout. "
            "Default: model-results."
        ),
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        nargs="+",
        help=(
            "One or more result directories to use instead of --experiment-preset. "
            "For example: --experiments-dir experiments/syntax_boundless"
        ),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Output directory for plots. Default: {DEFAULT_PLOTS_DIR}",
    )
    parser.add_argument(
        "--metric-file",
        default="test.uuas",
        help="Metric file to plot. Default: test.uuas.",
    )
    parser.add_argument(
        "--sentence-type",
        default="base_sentence",
        help="Sentence type folder to include. Default: base_sentence.",
    )
    parser.add_argument(
        "--attractor-count",
        type=int,
        default=4,
        help="Attractor count folder to include. Default: 4.",
    )
    parser.add_argument(
        "--task-name",
        default="parse-distance",
        help="Probe task folder to include. Default: parse-distance.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma", "qwen", "llama"],
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
    normalized = model.lower()
    gemma_match = re.search(r"gemma[-_]?(\d+(?:\.\d+)?)[-_](\d+)b", normalized)
    if gemma_match:
        return f"Gemma-{gemma_match.group(1)}-{gemma_match.group(2)}B"

    qwen_match = re.search(r"qwen[-_]?qwen?(\d+(?:\.\d+)?)[-_](\d+)b", normalized)
    if qwen_match:
        return f"Qwen-{qwen_match.group(1)}-{qwen_match.group(2)}B"

    llama_match = re.search(r"llama[-_]?(\d+(?:\.\d+)?)[-_](\d+)b", normalized)
    if llama_match:
        return f"Llama-{llama_match.group(1)}-{llama_match.group(2)}B"

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


def model_family(model):
    normalized = model.lower()
    if "gemma" in normalized:
        return "gemma"
    if "qwen" in normalized:
        return "qwen"
    if "olmo" in normalized:
        return "olmo"
    if "llama" in normalized:
        return "llama"
    return model.split("-", 1)[0].lower()


def model_sort_key(model):
    order = {"gemma": 0, "qwen": 1, "olmo": 2, "llama": 3}
    family = model_family(model)
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


def parse_syntax_boundless_metric_path(metric_path, experiments_dir):
    try:
        rel_parts = metric_path.relative_to(experiments_dir).parts
    except ValueError:
        return None

    attractor_index = next(
        (index for index, part in enumerate(rel_parts) if ATTRACTOR_RE.match(part)),
        None,
    )
    if attractor_index is None or attractor_index < 3:
        return None
    if len(rel_parts) <= attractor_index + 4:
        return None

    dataset_name = rel_parts[attractor_index - 3]
    variation = rel_parts[attractor_index - 2]
    sentence_type = rel_parts[attractor_index - 1]
    seed_name = rel_parts[0] if rel_parts[0].startswith("seed_") else "seed_42"
    attractor_dir = rel_parts[attractor_index]
    model_name = rel_parts[attractor_index + 1]
    layer_dir = rel_parts[attractor_index + 2]
    task_name = rel_parts[attractor_index + 3]

    attractor_match = ATTRACTOR_RE.match(attractor_dir)
    layer_match = LAYER_RE.search(layer_dir)
    if attractor_match is None or layer_match is None:
        return None

    try:
        value = read_metric_value(metric_path)
    except (OSError, ValueError):
        return None
    if value != value:
        return None

    return MetricPoint(
        dataset_name=dataset_name,
        variation=variation,
        sentence_type=sentence_type,
        attractor_count=int(attractor_match.group(1)),
        seed_name=seed_name,
        model_name=model_name,
        layer=int(layer_match.group(1)),
        task_name=task_name,
        metric_path=metric_path,
        modified_time=metric_path.stat().st_mtime,
        layout="syntax-boundless",
    )


def parse_flat_model_metric_path(metric_path, experiments_dir):
    try:
        rel_parts = metric_path.relative_to(experiments_dir).parts
    except ValueError:
        return None
    if len(rel_parts) < 3:
        return None

    result_dir = rel_parts[-2]
    result_match = FLAT_RESULT_RE.match(result_dir)
    if result_match is None:
        return None

    try:
        value = read_metric_value(metric_path)
    except (OSError, ValueError):
        return None
    if value != value:
        return None

    return MetricPoint(
        dataset_name="",
        variation="",
        sentence_type="",
        attractor_count=0,
        seed_name="default",
        model_name=experiments_dir.name,
        layer=int(result_match.group(1)),
        task_name=result_match.group(2),
        metric_path=metric_path,
        modified_time=metric_path.stat().st_mtime,
        layout="model-results",
    )


def read_metric_value(metric_path):
    return float(metric_path.read_text().strip())


def discover_points(experiments_dirs, metric_file):
    points = []
    for experiments_dir in experiments_dirs:
        parser = (
            parse_syntax_boundless_metric_path
            if experiments_dir.name == "syntax_boundless"
            else parse_flat_model_metric_path
        )
        for metric_path in experiments_dir.rglob(metric_file):
            point = parser(metric_path, experiments_dir)
            if point is not None:
                points.append(point)
    return points


def filter_points(points, args, model_filters):
    if points and all(point.layout == "model-results" for point in points):
        return [
            point
            for point in points
            if point.task_name == args.task_name
            and should_include_model(point.model_name, model_filters)
        ]

    required_inputs = {
        (dataset, variation)
        for hypothesis in HYPOTHESES.values()
        for dataset, variation in hypothesis["inputs"]
    }

    def matches(point):
        return (
            (point.dataset_name, point.variation) in required_inputs
            and point.sentence_type == args.sentence_type
            and point.attractor_count == args.attractor_count
            and point.task_name == args.task_name
            and should_include_model(point.model_name, model_filters)
        )

    return [point for point in points if matches(point)]


def keep_latest_per_layer(points):
    latest = {}
    for point in points:
        key = (
            point.dataset_name,
            point.variation,
            point.sentence_type,
            point.attractor_count,
            point.seed_name,
            point.model_name,
            point.layer,
            point.task_name,
        )
        previous = latest.get(key)
        if previous is None or point.modified_time > previous.modified_time:
            latest[key] = point
    return list(latest.values())


def layerwise_average(series_list):
    common_layers = sorted(set.intersection(*(set(series) for series in series_list)))
    averaged = {}
    for layer in common_layers:
        values = [series[layer] for series in series_list]
        averaged[layer] = sum(values) / len(values)
    return averaged


def aggregate_seed_series(seed_series):
    baseline_seed = "seed_42"
    if baseline_seed not in seed_series:
        return None

    common_layers = sorted(set.intersection(*(set(series) for series in seed_series.values())))
    baseline = {}
    standard_deviations = {}
    for layer in common_layers:
        values = [series[layer] for series in seed_series.values()]
        baseline[layer] = seed_series[baseline_seed][layer]
        standard_deviations[layer] = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": baseline, "std": standard_deviations}


def build_plot_data(points):
    if points and points[0].layout == "model-results":
        return {
            model: {
                "score": {
                    "mean": {
                        point.layer: read_metric_value(point.metric_path)
                        for point in model_points
                    },
                    "std": {},
                }
            }
            for model, model_points in group_points_by_model(points).items()
        }

    raw = {}
    for point in points:
        raw.setdefault(point.model_name, {}).setdefault(point.dataset_name, {}).setdefault(
            point.variation, {}
        ).setdefault(point.seed_name, {})[point.layer] = read_metric_value(point.metric_path)

    plot_data = {}
    for model, model_data in raw.items():
        model_plot_data = {}
        for hypothesis, config in HYPOTHESES.items():
            input_seed_series = []
            missing_inputs = []
            for dataset, variation in config["inputs"]:
                seed_series = model_data.get(dataset, {}).get(variation)
                if not seed_series:
                    missing_inputs.append(f"{dataset}/{variation}")
                else:
                    input_seed_series.append(seed_series)

            if missing_inputs:
                print(f"[skip] {model} {hypothesis}: missing {', '.join(missing_inputs)}")
                model_plot_data = {}
                break

            common_seeds = set.intersection(
                *(set(seed_series) for seed_series in input_seed_series)
            )
            if not common_seeds:
                print(f"[skip] {model} {hypothesis}: no common seeds")
                model_plot_data = {}
                break

            hypothesis_seed_series = {}
            for seed_name in common_seeds:
                series_list = [seed_series[seed_name] for seed_series in input_seed_series]
                series = (
                    layerwise_average(series_list)
                    if hypothesis == "syn"
                    else series_list[0]
                )
                if series:
                    hypothesis_seed_series[seed_name] = series

            if not hypothesis_seed_series:
                print(f"[skip] {model} {hypothesis}: no common layers")
                model_plot_data = {}
                break

            aggregated_series = aggregate_seed_series(hypothesis_seed_series)
            if aggregated_series is None:
                print(f"[skip] {model} {hypothesis}: missing seed-42 baseline")
                model_plot_data = {}
                break
            model_plot_data[hypothesis] = aggregated_series

        if model_plot_data:
            plot_data[model] = model_plot_data

    return plot_data


def group_points_by_model(points):
    grouped = {}
    for point in points:
        grouped.setdefault(point.model_name, []).append(point)
    return grouped


def as_percent(values):
    if not values:
        return values
    scale = 100 if max(values) <= 1.0 else 1
    return [value * scale for value in values]


def plot_models(plot_data, plots_dir, formats, metric_file):
    models = sorted(plot_data, key=model_sort_key)
    if not models:
        return []

    raw_values = [
        value
        for model_data in plot_data.values()
        for series in model_data.values()
        for value in series["mean"].values()
    ]
    value_scale = 100 if max(raw_values) <= 1.0 else 1
    largest_value = max(
        (mean + std) * value_scale
        for model_data in plot_data.values()
        for series in model_data.values()
        for layer, mean in series["mean"].items()
        for std in [series["std"].get(layer, 0.0)]
    )
    y_tick_step = 5 if largest_value <= 40 else 10
    y_upper_limit = max(20, math.ceil(largest_value / y_tick_step) * y_tick_step)
    y_ticks = list(range(0, int(y_upper_limit) + 1, y_tick_step))

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(5 * len(models), 4.8),
        sharey=True,
    )
    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        all_layers = sorted(
            set().union(*(series["mean"].keys() for series in plot_data[model].values()))
        )

        series_configs = (
            HYPOTHESES.items()
            if "score" not in plot_data[model]
            else [("score", {"label": "Test UUAS"})]
        )
        for hypothesis, config in series_configs:
            series = plot_data[model][hypothesis]
            layers = sorted(series["mean"])
            values = [series["mean"][layer] * value_scale for layer in layers]
            standard_deviations = [
                series["std"].get(layer, 0.0) * value_scale for layer in layers
            ]
            color = COLORS.get(hypothesis, "#3C5488")

            ax.plot(
                layers,
                values,
                marker=MARKERS.get(hypothesis, "o"),
                color=color,
                linewidth=2.5,
                markersize=9,
                label=config["label"],
            )
            if any(standard_deviations):
                ax.fill_between(
                    layers,
                    [value - std for value, std in zip(values, standard_deviations)],
                    [value + std for value, std in zip(values, standard_deviations)],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

        ax.set_ylim(0, y_upper_limit)
        ax.set_yticks(y_ticks)
        if all_layers:
            ax.set_xlim(min(all_layers) - 2, max(all_layers) + 2)
        ax.set_xticks(
            [layer for layer in [0, 10, 20, 30, 40] if not all_layers or layer <= max(all_layers)]
        )
        ax.set_xlabel(f"Layer Index\n\n({chr(97 + idx)}) {format_model_name(model)}")
        ax.yaxis.grid(True, linestyle="-", alpha=0.2)

        if idx == 0:
            ax.set_ylabel("Test UUAS (%)")

    is_hypothesis_plot = "score" not in plot_data[models[0]]
    if is_hypothesis_plot:
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
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        plt.subplots_adjust(wspace=0.08, bottom=0.42)
    else:
        plt.subplots_adjust(wspace=0.08, bottom=0.22)

    model_suffix = "_".join(model_family(model) for model in models)
    model_suffix = "_".join(dict.fromkeys(model_suffix.split("_")))
    metric_stem = metric_file.replace(".", "_")

    saved_paths = []
    for plot_format in formats:
        suffix = plot_format.lower().lstrip(".")
        save_path = plots_dir / f"{model_suffix}_{metric_stem}_hypotheses_lineplot.{suffix}"
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        saved_paths.append(save_path)

    plt.close(fig)
    return saved_paths


def main():
    args = parse_args()
    raw_experiments_dirs = args.experiments_dir or EXPERIMENT_PRESETS[args.experiment_preset]
    experiments_dirs = [path.expanduser().resolve() for path in raw_experiments_dirs]
    plots_dir = args.plots_dir.expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    missing_dirs = [path for path in experiments_dirs if not path.is_dir()]
    if missing_dirs:
        print("Experiment directories not found:")
        for path in missing_dirs:
            print(f"  {path}")
        return

    model_filters = normalize_model_filters(args.models)
    if model_filters is None:
        print("[models] including all models")
    else:
        print(f"[models] including: {', '.join(model_filters)}")

    points = discover_points(experiments_dirs, args.metric_file)
    points = filter_points(points, args, model_filters)
    points = keep_latest_per_layer(points)

    if not points:
        print(
            f"No matching {args.metric_file} values found under "
            f"{', '.join(str(path) for path in experiments_dirs)}"
        )
        return

    configure_plot_style()
    plot_data = build_plot_data(points)
    if not plot_data:
        print("No complete model data found to plot.")
        return

    saved_paths = plot_models(plot_data, plots_dir, args.formats, args.metric_file)
    for save_path in saved_paths:
        print(f"Generated probing plot at {save_path}")


if __name__ == "__main__":
    main()
