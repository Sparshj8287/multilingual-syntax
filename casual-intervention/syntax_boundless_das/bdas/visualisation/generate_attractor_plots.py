import argparse
import json
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
DEFAULT_RESULTS_DIR = VIS_DIR.parent / "results_dbm"
DEFAULT_PLOTS_DIR = VIS_DIR / "plots_dbm_attractors"

ATTRACTORS = [1, 2, 3, 4]
DIRECTIONS = {
    "source_to_base": {
        "setup": "SG $\\rightarrow$ PL",
        "filename": "sg_to_pl",
    },
    "base_to_source": {
        "setup": "PL $\\rightarrow$ SG",
        "filename": "pl_to_sg",
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

# Keep colors consistent with generate_lineplots.py.
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
            "Generate Boundless DAS attractor-complexity plots. "
            "For each attractor count, syntax uses the peak of the layer-wise "
            "macro-average across linear and bow; proximity and majority use "
            "their peak layer accuracy on their corresponding datasets."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"DAS results directory containing model/dataset/variation folders. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Output directory for attractor plots. Default: {DEFAULT_PLOTS_DIR}",
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


def parse_results_tree(results_dir, model_filters):
    data = {}

    for summary_path in results_dir.rglob("all_layers_summary.json"):
        parts = summary_path.relative_to(results_dir).parts
        if len(parts) < 6:
            continue

        model = parts[0]
        dataset = parts[1]
        variation = parts[2]
        attractor_dir = parts[3]
        direction_idx = 4

        if not should_include_model(model, model_filters):
            continue

        try:
            attractor = int(attractor_dir.split("_", 1)[0])
        except ValueError:
            continue

        if attractor not in ATTRACTORS:
            continue

        if parts[direction_idx] not in DIRECTIONS:
            direction_idx += 1
            if len(parts) <= direction_idx or parts[direction_idx] not in DIRECTIONS:
                continue

        direction = parts[direction_idx]
        data.setdefault(model, {}).setdefault(direction, {}).setdefault(attractor, {}).setdefault(
            dataset, {}
        ).setdefault(variation, {})

        with summary_path.open("r") as f:
            summary = json.load(f)

        for layer_info in summary.get("layers", []):
            layer = layer_info.get("layer")
            accuracy = layer_info.get("final_test_accuracy")
            if layer is None or accuracy is None:
                continue
            data[model][direction][attractor][dataset][variation][int(layer)] = float(accuracy)

    return data


def layerwise_average(series_list):
    layers = sorted(set.intersection(*(set(series.keys()) for series in series_list)))
    averaged = {}
    for layer in layers:
        values = [series[layer] for series in series_list]
        averaged[layer] = sum(values) / len(values)
    return averaged


def peak_accuracy_for_hypothesis(direction_data, attractor, hypothesis):
    series_list = []
    for dataset, variation in HYPOTHESES[hypothesis]["inputs"]:
        layers = direction_data.get(attractor, {}).get(dataset, {}).get(variation)
        if not layers:
            raise KeyError(f"Missing DAS data for {attractor}_attractor/{dataset}/{variation}")
        series_list.append(layers)

    if hypothesis == "syn":
        # Peak of the layer-wise macro-average, not the average of independent peaks.
        peak_series = layerwise_average(series_list)
    else:
        peak_series = series_list[0]

    if not peak_series:
        raise ValueError(f"No layer accuracies for {attractor}_attractor/{hypothesis}")

    return max(peak_series.values())


def build_plot_data(data):
    plot_data = {}
    for model, model_data in data.items():
        model_plot_data = {}
        for direction in DIRECTIONS:
            direction_data = model_data.get(direction, {})
            direction_plot_data = {}
            direction_complete = True

            for hypothesis in HYPOTHESES:
                peaks = {}
                for attractor in ATTRACTORS:
                    try:
                        peaks[attractor] = peak_accuracy_for_hypothesis(
                            direction_data, attractor, hypothesis
                        )
                    except (KeyError, ValueError) as error:
                        print(f"[skip] {model} {direction} {hypothesis}: {error}")
                        direction_complete = False
                        break

                if not direction_complete:
                    break
                direction_plot_data[hypothesis] = peaks

            if direction_complete and direction_plot_data:
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
        ax = axes[idx]
        direction_config = DIRECTIONS[direction]
        panel_data = plot_data[model][direction]

        for hypothesis in HYPOTHESES:
            peaks = panel_data[hypothesis]
            values = [peaks.get(attractor, float("nan")) * 100 for attractor in ATTRACTORS]
            ax.plot(
                ATTRACTORS,
                values,
                marker=MARKERS[hypothesis],
                color=COLORS[hypothesis],
                linewidth=3,
                markersize=11,
            )

        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, 4.5)
        ax.set_xticks(ATTRACTORS)
        panel_label = f"({chr(97 + idx)}) {format_model_name(model)} ({direction_config['setup']})"
        ax.set_xlabel(f"Number of Attractors\n\n{panel_label}")
        ax.yaxis.grid(True, linestyle="-", alpha=0.2)

        if idx == 0:
            ax.set_ylabel("Peak IIA (%)")
            ax.set_yticks([0, 25, 50, 75, 100])

    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            color=COLORS[hypothesis],
            marker=MARKERS[hypothesis],
            linewidth=3,
            markersize=11,
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

    model_suffix = "_".join(model.split("-", 1)[0].lower() for model, _direction in panels)
    model_suffix = "_".join(dict.fromkeys(model_suffix.split("_")))

    saved_paths = []
    for plot_format in formats:
        suffix = plot_format.lower().lstrip(".")
        save_path = plots_dir / f"{model_suffix}_peak_iia_by_attractor.{suffix}"
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        saved_paths.append(save_path)

    plt.close(fig)
    return saved_paths


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    plots_dir = args.plots_dir.expanduser().resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_filters = normalize_model_filters(args.models)
    if model_filters is None:
        print("[models] including all models")
    else:
        print(f"[models] including: {', '.join(model_filters)}")

    data = parse_results_tree(results_dir, model_filters)
    if not data:
        print(f"No all_layers_summary.json files found in {results_dir}")
        return

    configure_plot_style()
    plot_data = build_plot_data(data)
    if not plot_data:
        print("No complete model/setup combinations found to plot.")
        return

    saved_paths = plot_combined_models(plot_data, plots_dir, args.formats)
    for save_path in saved_paths:
        print(f"Generated attractor plot at {save_path}")


if __name__ == "__main__":
    main()
