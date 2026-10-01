#!/usr/bin/env python3
import argparse
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
COLORS = {
    "syn": "#00A087",
    "prox": "#E64B35",
    "maj": "#3C5488",
}
LABELS = {
    "syn": r"$\mathcal{H}_{syn}$ (Syntax)",
    "prox": r"$\mathcal{H}_{prox}$ (Proximity)",
    "maj": r"$\mathcal{H}_{maj}$ (Majority)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot accuracy for the syntax, proximity, and majority hypotheses."
    )
    parser.add_argument("--syn", type=float, default=97.79)
    parser.add_argument("--prox", type=float, default=98.93)
    parser.add_argument("--maj", type=float, default=98.95)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    return parser.parse_args()


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
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    hypotheses = ["syn", "prox", "maj"]
    values = [args.syn, args.prox, args.maj]

    x_positions = [0, 0.7, 1.4]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(
        x_positions,
        values,
        width=0.22,
        color=[COLORS[hypothesis] for hypothesis in hypotheses],
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(
        x_positions,
        [LABELS[hypothesis] for hypothesis in hypotheses],
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        output_path = args.output_dir / f"hypothesis_accuracy.{suffix}"
        fig.savefig(output_path, dpi=500, bbox_inches="tight")
        print(f"[saved] {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
