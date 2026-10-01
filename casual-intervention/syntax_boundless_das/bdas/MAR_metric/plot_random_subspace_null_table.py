#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs_null" / "mar_random_subspace_null_stats.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_null"

MODEL_LABELS = {
    "gemma-3-12b-pt": "Gemma-3-12B",
    "qwen-3-8b": "Qwen-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "olmo-3-7b": "OLMo-3-7B",
}
DIRECTION_LABELS = {
    "base_to_source": r"PL $\rightarrow$ SG",
    "source_to_base": r"SG $\rightarrow$ PL",
}
COMPARISON_LABELS = {
    "syntax_linear_vs_prox": r"$\mathcal{H}_{prox}$",
    "syntax_bow_vs_maj": r"$\mathcal{H}_{maj}$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render random-subspace MAR null statistics as a publication table."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def format_p_value(value: float) -> str:
    if value < 1e-10:
        return r"$< 10^{-10}$"
    if value < 0.001:
        exponent = int(f"{value:.0e}".split("e")[1])
        coefficient = value / (10**exponent)
        return rf"${coefficient:.2f} \times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = ["Model", "Direction", "Comparison", r"Null Mean $\pm$ Std", "Z-score", "P-value"]
    cell_text = []
    for row in rows:
        cell_text.append(
            [
                MODEL_LABELS.get(row["model"], row["model"]),
                DIRECTION_LABELS.get(row["direction"], row["direction"]),
                COMPARISON_LABELS.get(row["comparison"], row["comparison"]),
                rf"${row['null_mean']:.4f} \pm {row['null_std']:.4f}$",
                f"{row['z_score']:.2f}",
                format_p_value(row["right_tailed_p_value"]),
            ]
        )

    figure_height = 0.34 * (len(rows) + 1) + 0.35
    fig, ax = plt.subplots(figsize=(8.2, figure_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        colLoc="center",
        cellLoc="left",
        colWidths=[0.19, 0.16, 0.16, 0.22, 0.13, 0.14],
        bbox=[0.015, 0.02, 0.97, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    row_count = len(rows)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.PAD = 0.08
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row_idx == row_count:
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        else:
            cell.visible_edges = ""
        if col_idx >= 3:
            cell.get_text().set_ha("center")

    for row_idx in range(2, row_count + 1):
        previous_model = rows[row_idx - 2]["model"]
        current_model = rows[row_idx - 1]["model"]
        if current_model != previous_model:
            for col_idx in range(len(columns)):
                cell = table[(row_idx, col_idx)]
                cell.visible_edges = "T" + ("B" if row_idx == row_count else "")
                cell.set_linewidth(0.6)

    for suffix in ("png", "pdf"):
        output_path = args.output_dir / f"mar_random_subspace_null_table.{suffix}"
        fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
        print(f"[saved] {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
