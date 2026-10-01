#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt


VIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = VIS_DIR / "tables_das_seeds" / "das_iia_seed_mean_std_decimal.json"
DEFAULT_OUTPUT_DIR = VIS_DIR / "tables_das_seeds"

MODEL_LABELS = {
    "gemma-3-12b-pt": "Gemma-3-12B",
    "qwen-3-8b": "Qwen-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "olmo-3-7b": "OLMo-3-7B",
}
MODEL_ORDER = ["gemma-3-12b-pt", "qwen-3-8b", "llama-3.1-8b", "olmo-3-7b"]
DIRECTION_LABELS = {
    "base_to_source": r"PL $\rightarrow$ SG",
    "source_to_base": r"SG $\rightarrow$ PL",
}
DIRECTION_ORDER = ["source_to_base", "base_to_source"]
HYPOTHESIS_LABELS = {
    "syn": r"$\mathcal{H}_{syn}$",
    "prox": r"$\mathcal{H}_{prox}$",
    "maj": r"$\mathcal{H}_{maj}$",
}
HYPOTHESIS_COLORS = {
    "syn": "#00A087",
    "prox": "#E64B35",
    "maj": "#3C5488",
}
HYPOTHESIS_ORDER = ["syn", "prox", "maj"]
INCOMPLETE_MARKER = r"$^{\dagger}$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the Boundless DAS seed mean +/- standard deviation table as a "
            "publication-style PDF using decimal IIA values."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model directory names or aliases to include. Default: all.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Write only the combined table and skip the per-model PDFs.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
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


def cell_value(rows_by_key, model, direction, layer, hypothesis):
    row = rows_by_key.get((model, direction, layer, hypothesis))
    if row is None:
        return "--"
    marker = INCOMPLETE_MARKER if row["n_seeds"] < 3 else ""
    return rf"${row['mean']:.4f}\,\pm\,{row['std']:.4f}${marker}"


def build_cell_text(rows, models, include_model_column):
    rows_by_key = {
        (row["model"], row["direction"], row["layer"], row["hypothesis"]): row
        for row in rows
    }

    cell_text = []
    group_starts = []
    for model in models:
        for direction in DIRECTION_ORDER:
            layers = sorted(
                {
                    row["layer"]
                    for row in rows
                    if row["model"] == model and row["direction"] == direction
                }
            )
            if not layers:
                continue
            group_starts.append(len(cell_text))
            for layer in layers:
                record = [DIRECTION_LABELS[direction], str(layer)]
                if include_model_column:
                    record.insert(0, MODEL_LABELS.get(model, model))
                record.extend(
                    cell_value(rows_by_key, model, direction, layer, hypothesis)
                    for hypothesis in HYPOTHESIS_ORDER
                )
                cell_text.append(record)
    return cell_text, group_starts


def render_table(cell_text, group_starts, columns, col_widths, width, output_stem, has_marker):
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
            (
                r"$^{\dagger}$Computed from seeds 96 and 1120 only; "
                "results_das_full has no seed-42 run for that cell."
            ),
            fontsize=9,
            ha="left",
            va="bottom",
        )

    saved = []
    for suffix in ("pdf", "png"):
        output_path = output_stem.parent / f"{output_stem.name}.{suffix}"
        fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
        saved.append(output_path)
        print(f"[saved] {output_path}")
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text())
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    requested = [model.strip().lower() for raw in args.models for model in raw.split(",")]
    available = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
    if "all" not in requested:
        available = [
            model
            for model in available
            if any(token in model for token in requested)
        ]
    if not available:
        raise SystemExit("No matching models found.")

    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hypothesis_columns = [HYPOTHESIS_LABELS[hypothesis] for hypothesis in HYPOTHESIS_ORDER]

    cell_text, group_starts = build_cell_text(rows, available, include_model_column=True)
    render_table(
        cell_text,
        group_starts,
        ["Model", "Direction", "Layer", *hypothesis_columns],
        [0.16, 0.13, 0.08, 0.21, 0.21, 0.21],
        9.6,
        args.output_dir / "das_iia_seed_mean_std_table",
        any(INCOMPLETE_MARKER in cell for row in cell_text for cell in row),
    )

    if args.combined_only:
        return

    for model in available:
        model_cells, model_groups = build_cell_text(rows, [model], include_model_column=False)
        render_table(
            model_cells,
            model_groups,
            ["Direction", "Layer", *hypothesis_columns],
            [0.15, 0.09, 0.2533, 0.2533, 0.2533],
            8.6,
            args.output_dir / f"das_iia_seed_mean_std_table_{model}",
            any(INCOMPLETE_MARKER in cell for row in model_cells for cell in row),
        )


if __name__ == "__main__":
    main()
