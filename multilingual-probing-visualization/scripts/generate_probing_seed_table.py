#!/usr/bin/env python3
"""Build a per-layer mean +/- standard deviation table of probing UUAS across seeds.

Seed 42 lives directly under experiments/syntax_boundless/<dataset>, while seed 96
and seed 1120 live under experiments/syntax_boundless/seed_<n>/<dataset>. Unlike the
Boundless DAS results there is no intervention direction here.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt

from generate_probing_lineplots import (
    HYPOTHESES,
    format_model_name,
    keep_latest_per_layer,
    layerwise_average,
    model_family,
    model_sort_key,
    normalize_model_filters,
    parse_syntax_boundless_metric_path,
    read_metric_value,
    should_include_model,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_EXPERIMENTS_DIR = PROJECT_DIR / "experiments" / "syntax_boundless"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "visualizations_plots" / "probing_seed_tables"

BASELINE_SEED = "seed_42"
SWEEP_SEEDS = ("seed_96", "seed_1120")
ALL_SEEDS = (BASELINE_SEED, *SWEEP_SEEDS)

HYPOTHESIS_ORDER = ["syn", "prox", "maj"]
HYPOTHESIS_HEADERS = {
    "syn": r"$\mathcal{H}_{syn}$",
    "prox": r"$\mathcal{H}_{prox}$",
    "maj": r"$\mathcal{H}_{maj}$",
}
HYPOTHESIS_PLAIN_LABELS = {
    "syn": "H_syn (Syntax)",
    "prox": "H_prox (Proximity)",
    "maj": "H_maj (Majority)",
}
HYPOTHESIS_COLORS = {
    "syn": "#00A087",
    "prox": "#E64B35",
    "maj": "#3C5488",
}
INCOMPLETE_MARKER = r"$^{\dagger}$"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-layer mean +/- standard deviation table of probing test UUAS "
            f"across seeds {', '.join(ALL_SEEDS)}."
        )
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help=f"syntax_boundless results directory. Default: {DEFAULT_EXPERIMENTS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the CSV, JSON, and table files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma", "qwen", "llama"],
        help="Model aliases to include. Use --models all for every model.",
    )
    parser.add_argument(
        "--metric-file",
        default="test.uuas",
        help="Metric file to aggregate. Default: test.uuas.",
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
        "--percent",
        action="store_true",
        help="Report UUAS as percentages instead of decimals.",
    )
    parser.add_argument(
        "--per-model",
        action="store_true",
        help="Also write one table per model alongside the combined table.",
    )
    return parser.parse_args()


def configure_table_style():
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
            "font.size": 11,
            "mathtext.fontset": "stix",
        }
    )


def discover_points(experiments_dir, args, model_filters):
    required_inputs = {
        (dataset, variation)
        for hypothesis in HYPOTHESES.values()
        for dataset, variation in hypothesis["inputs"]
    }

    points = []
    for metric_path in experiments_dir.rglob(args.metric_file):
        point = parse_syntax_boundless_metric_path(metric_path, experiments_dir)
        if point is None:
            continue
        if (point.dataset_name, point.variation) not in required_inputs:
            continue
        if point.sentence_type != args.sentence_type:
            continue
        if point.attractor_count != args.attractor_count:
            continue
        if point.task_name != args.task_name:
            continue
        if point.seed_name not in ALL_SEEDS:
            continue
        if not should_include_model(point.model_name, model_filters):
            continue
        points.append(point)
    return keep_latest_per_layer(points)


def build_seed_series(points):
    """Map model -> hypothesis -> seed -> {layer: value}."""
    raw = {}
    for point in points:
        raw.setdefault(point.model_name, {}).setdefault(point.dataset_name, {}).setdefault(
            point.variation, {}
        ).setdefault(point.seed_name, {})[point.layer] = read_metric_value(point.metric_path)

    series_by_model = {}
    for model, model_data in raw.items():
        hypothesis_series = {}
        for hypothesis in HYPOTHESIS_ORDER:
            input_seed_series = []
            missing = []
            for dataset, variation in HYPOTHESES[hypothesis]["inputs"]:
                seed_series = model_data.get(dataset, {}).get(variation)
                if not seed_series:
                    missing.append(f"{dataset}/{variation}")
                else:
                    input_seed_series.append(seed_series)
            if missing:
                print(f"[skip] {model} {hypothesis}: missing {', '.join(missing)}")
                continue

            per_seed = {}
            for seed in ALL_SEEDS:
                series_list = [
                    seed_series[seed]
                    for seed_series in input_seed_series
                    if seed in seed_series
                ]
                if len(series_list) != len(input_seed_series):
                    continue
                # H_syn averages the linear and bow runs of obj_rel_across_anim layer by layer.
                series = layerwise_average(series_list) if len(series_list) > 1 else series_list[0]
                if series:
                    per_seed[seed] = series
            if per_seed:
                hypothesis_series[hypothesis] = per_seed
        if hypothesis_series:
            series_by_model[model] = hypothesis_series
    return series_by_model


def build_rows(series_by_model):
    rows = []
    for model in sorted(series_by_model, key=model_sort_key):
        hypothesis_series = series_by_model[model]
        layers = sorted(
            {
                layer
                for per_seed in hypothesis_series.values()
                for series in per_seed.values()
                for layer in series
            }
        )
        for hypothesis in HYPOTHESIS_ORDER:
            per_seed = hypothesis_series.get(hypothesis)
            if per_seed is None:
                continue
            for layer in layers:
                seed_values = {
                    seed: series[layer]
                    for seed, series in per_seed.items()
                    if layer in series
                }
                if not seed_values:
                    continue
                values = list(seed_values.values())
                rows.append(
                    {
                        "model": model,
                        "model_label": format_model_name(model),
                        "hypothesis": hypothesis,
                        "hypothesis_label": HYPOTHESIS_PLAIN_LABELS[hypothesis],
                        "layer": layer,
                        "n_seeds": len(values),
                        "seeds": ",".join(sorted(seed_values, key=ALL_SEEDS.index)),
                        "mean": statistics.fmean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        **{seed: seed_values.get(seed) for seed in ALL_SEEDS},
                    }
                )
    return rows


def scale_rows(rows, scale):
    if scale == 1:
        return rows
    scaled = []
    for row in rows:
        scaled_row = dict(row)
        for field in ("mean", "std", *ALL_SEEDS):
            if scaled_row.get(field) is not None:
                scaled_row[field] = scaled_row[field] * scale
        scaled.append(scaled_row)
    return scaled


def write_outputs(rows, output_dir, percent):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "percent" if percent else "decimal"

    csv_path = output_dir / f"probing_uuas_seed_mean_std_{suffix}.csv"
    fieldnames = [
        "model",
        "model_label",
        "hypothesis",
        "hypothesis_label",
        "layer",
        "n_seeds",
        "seeds",
        "mean",
        "std",
        *ALL_SEEDS,
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )

    json_path = output_dir / f"probing_uuas_seed_mean_std_{suffix}.json"
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def print_table(rows, percent):
    digits = 2 if percent else 4
    for model in dict.fromkeys(row["model"] for row in rows):
        panel = [row for row in rows if row["model"] == model]
        print(f"\n{format_model_name(model)}")
        header = f"{'Layer':>6}" + "".join(
            f"{HYPOTHESIS_PLAIN_LABELS[hypothesis]:>28}" for hypothesis in HYPOTHESIS_ORDER
        )
        print(header)
        for layer in sorted({row["layer"] for row in panel}):
            cells = []
            for hypothesis in HYPOTHESIS_ORDER:
                match = next(
                    (
                        row
                        for row in panel
                        if row["hypothesis"] == hypothesis and row["layer"] == layer
                    ),
                    None,
                )
                if match is None:
                    cells.append(f"{'--':>28}")
                else:
                    marker = "*" if match["n_seeds"] < len(ALL_SEEDS) else ""
                    text = (
                        f"{match['mean']:.{digits}f} \u00b1 {match['std']:.{digits}f}{marker}"
                    )
                    cells.append(f"{text:>28}")
            print(f"{layer:>6}" + "".join(cells))


def cell_text_for(rows_by_key, model, layer, hypothesis, digits):
    row = rows_by_key.get((model, layer, hypothesis))
    if row is None:
        return "--"
    marker = INCOMPLETE_MARKER if row["n_seeds"] < len(ALL_SEEDS) else ""
    return rf"${row['mean']:.{digits}f}\,\pm\,{row['std']:.{digits}f}${marker}"


def build_cell_text(rows, models, include_model_column, digits):
    rows_by_key = {
        (row["model"], row["layer"], row["hypothesis"]): row for row in rows
    }

    cell_text = []
    group_starts = []
    for model in models:
        layers = sorted({row["layer"] for row in rows if row["model"] == model})
        if not layers:
            continue
        group_starts.append(len(cell_text))
        for layer in layers:
            record = [str(layer)]
            if include_model_column:
                record.insert(0, format_model_name(model))
            record.extend(
                cell_text_for(rows_by_key, model, layer, hypothesis, digits)
                for hypothesis in HYPOTHESIS_ORDER
            )
            cell_text.append(record)
    return cell_text, group_starts


def render_table(cell_text, group_starts, columns, col_widths, width, output_stem):
    has_marker = any(INCOMPLETE_MARKER in cell for row in cell_text for cell in row)
    figure_height = 0.30 * (len(cell_text) + 1) + (0.55 if has_marker else 0.2)
    fig, ax = plt.subplots(figsize=(width, figure_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        colLoc="center",
        cellLoc="left",
        colWidths=col_widths,
        bbox=[0.015, 0.02, 0.97, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    value_start = len(columns) - len(HYPOTHESIS_ORDER)
    row_count = len(cell_text)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.PAD = 0.08
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
            if col_idx >= value_start:
                cell.get_text().set_color(
                    HYPOTHESIS_COLORS[HYPOTHESIS_ORDER[col_idx - value_start]]
                )
        elif row_idx == row_count:
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        else:
            cell.visible_edges = ""
        if col_idx >= value_start - 1:
            cell.get_text().set_ha("center")

    for start in group_starts[1:]:
        for col_idx in range(len(columns)):
            cell = table[(start + 1, col_idx)]
            cell.visible_edges = "T" + ("B" if start + 1 == row_count else "")
            cell.set_linewidth(0.6)

    if has_marker:
        fig.text(
            0.015,
            0.004,
            rf"$^{{\dagger}}$Computed from fewer than {len(ALL_SEEDS)} seeds.",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    for suffix in ("pdf", "png"):
        output_path = output_stem.parent / f"{output_stem.name}.{suffix}"
        fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
        print(f"[saved] {output_path}")
    plt.close(fig)


def main():
    args = parse_args()
    experiments_dir = args.experiments_dir.expanduser().resolve()
    if not experiments_dir.is_dir():
        raise SystemExit(f"Experiments directory not found: {experiments_dir}")

    model_filters = normalize_model_filters(args.models)
    points = discover_points(experiments_dir, args, model_filters)
    if not points:
        raise SystemExit(
            f"No attractor-{args.attractor_count} {args.metric_file} values found under "
            f"{experiments_dir}"
        )

    rows = scale_rows(build_rows(build_seed_series(points)), 100 if args.percent else 1)
    if not rows:
        raise SystemExit("No hypothesis series could be assembled.")

    print_table(rows, args.percent)

    incomplete = sorted(
        {
            (row["model_label"], row["hypothesis_label"], row["seeds"])
            for row in rows
            if row["n_seeds"] < len(ALL_SEEDS)
        }
    )
    if incomplete:
        print(f"\n[warn] cells computed from fewer than {len(ALL_SEEDS)} seeds:")
        for model_label, hypothesis_label, seeds in incomplete:
            print(f"  - {model_label} {hypothesis_label}: {seeds}")

    output_dir = args.output_dir.expanduser().resolve()
    csv_path, json_path = write_outputs(rows, output_dir, args.percent)
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {json_path}")

    configure_table_style()
    digits = 2 if args.percent else 4
    hypothesis_columns = [HYPOTHESIS_HEADERS[hypothesis] for hypothesis in HYPOTHESIS_ORDER]
    models = sorted({row["model"] for row in rows}, key=model_sort_key)

    cell_text, group_starts = build_cell_text(rows, models, True, digits)
    render_table(
        cell_text,
        group_starts,
        ["Model", "Layer", *hypothesis_columns],
        [0.18, 0.09, 0.243, 0.243, 0.243],
        8.8,
        output_dir / "probing_uuas_seed_mean_std_table",
    )

    if not args.per_model:
        return

    for model in models:
        model_cells, model_groups = build_cell_text(rows, [model], False, digits)
        render_table(
            model_cells,
            model_groups,
            ["Layer", *hypothesis_columns],
            [0.1, 0.3, 0.3, 0.3],
            8.0,
            output_dir / f"probing_uuas_seed_mean_std_table_{model_family(model)}",
        )


if __name__ == "__main__":
    main()
