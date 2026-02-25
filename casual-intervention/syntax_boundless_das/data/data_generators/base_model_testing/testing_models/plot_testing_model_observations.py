#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

models_to_use= "it"

@dataclass
class ResultPoint:
    nua: int
    base_acc: float
    source_acc: float
    discarded_f1: int
    discarded_f2: int
    valid_tested: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot testing_models observation summaries for linear/bow/maximal with "
            "separate main/aux figures and discard stats in legend."
        )
    )
    parser.add_argument(
        "--observations-dir",
        default=(
            "/home/sparsh/projects/multilingual-syntax/casual-intervention/"
            "syntax_boundless_das/data/data_generators/base_model_testing/"
            "testing_models/observations"
        ),
        help="Path to testing_models observations directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["linear", "bow", "maximal"],
        help="Dataset folders to plot.",
    )
    parser.add_argument(
        "--plots-subdir",
        default="plots",
        help="Subdirectory under observations-dir to save plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def extract_nua_from_stem(stem: str) -> int:
    match = re.search(r"_(\d+)$", stem)
    if not match:
        raise ValueError(f"Could not parse N attractors from filename stem: {stem}")
    return int(match.group(1))


def parse_summary_file(path: Path) -> dict:
    out = {
        "discarded_filter1_multitoken": None,
        "discarded_filter2_baseline_fail_main": None,
        "discarded_filter2_baseline_fail_aux": None,
        "valid_tested_entries_main": None,
        "valid_tested_entries_aux": None,
        "main_base_overall_accuracy": None,
        "main_source_overall_accuracy": None,
        "aux_base_overall_accuracy": None,
        "aux_source_overall_accuracy": None,
    }

    section = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line == "--- Main Verb Results ---":
                section = "main"
                continue
            if line == "--- Auxiliary Verbs Results ---":
                section = "aux"
                continue
            if line.startswith("--- Filter 2 Baseline Samples"):
                # Summary header is finished by this point.
                break

            if line.startswith("discarded_filter1_multitoken:"):
                out["discarded_filter1_multitoken"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("discarded_filter2_baseline_fail_main:"):
                out["discarded_filter2_baseline_fail_main"] = int(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("discarded_filter2_baseline_fail_aux:"):
                out["discarded_filter2_baseline_fail_aux"] = int(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("valid_tested_entries_main:"):
                out["valid_tested_entries_main"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("valid_tested_entries_aux:"):
                out["valid_tested_entries_aux"] = int(line.split(":", 1)[1].strip())
            elif section == "main" and line.startswith("base_overall_accuracy:"):
                out["main_base_overall_accuracy"] = float(line.split(":", 1)[1].strip())
            elif section == "main" and line.startswith("source_overall_accuracy:"):
                out["main_source_overall_accuracy"] = float(
                    line.split(":", 1)[1].strip()
                )
            elif section == "aux" and line.startswith("base_overall_accuracy:"):
                out["aux_base_overall_accuracy"] = float(line.split(":", 1)[1].strip())
            elif section == "aux" and line.startswith("source_overall_accuracy:"):
                out["aux_source_overall_accuracy"] = float(
                    line.split(":", 1)[1].strip()
                )

    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise ValueError(f"Missing required fields in {path}: {missing}")
    return out


def collect_dataset_points(
    observations_dir: Path, dataset: str
) -> tuple[dict[str, list[ResultPoint]], dict[str, list[ResultPoint]]]:
    main_points: dict[str, list[ResultPoint]] = {}
    aux_points: dict[str, list[ResultPoint]] = {}

    for model_dir in sorted(observations_dir.iterdir()):

        if not model_dir.name.endswith(models_to_use):
            continue
        if not model_dir.is_dir() or model_dir.name == "plots":
            continue

        dataset_dir = model_dir / dataset
        if not dataset_dir.exists():
            continue

        for txt_file in sorted(dataset_dir.glob("*.txt")):
            nua = extract_nua_from_stem(txt_file.stem)
            parsed = parse_summary_file(txt_file)

            main_points.setdefault(model_dir.name, []).append(
                ResultPoint(
                    nua=nua,
                    base_acc=parsed["main_base_overall_accuracy"],
                    source_acc=parsed["main_source_overall_accuracy"],
                    discarded_f1=parsed["discarded_filter1_multitoken"],
                    discarded_f2=parsed["discarded_filter2_baseline_fail_main"],
                    valid_tested=parsed["valid_tested_entries_main"],
                )
            )
            aux_points.setdefault(model_dir.name, []).append(
                ResultPoint(
                    nua=nua,
                    base_acc=parsed["aux_base_overall_accuracy"],
                    source_acc=parsed["aux_source_overall_accuracy"],
                    discarded_f1=parsed["discarded_filter1_multitoken"],
                    discarded_f2=parsed["discarded_filter2_baseline_fail_aux"],
                    valid_tested=parsed["valid_tested_entries_aux"],
                )
            )

    for points_by_model in (main_points, aux_points):
        for model_name in points_by_model:
            points_by_model[model_name].sort(key=lambda p: p.nua)

    return main_points, aux_points


def _series_text(points: list[ResultPoint], attr: str) -> str:
    items = [f"{p.nua}:{getattr(p, attr)}" for p in points]
    return "{" + ", ".join(items) + "}"


def make_legend_label(
    model_name: str, points: list[ResultPoint], metric_name: str
) -> str:
    f2_text = _series_text(points, "discarded_f2")
    valid_text = _series_text(points, "valid_tested")
    if metric_name.lower().startswith("aux"):
        return f"{model_name} (discard_f2={f2_text}, valid={valid_text})"
    f1_text = _series_text(points, "discarded_f1")
    return (
        f"{model_name} (discard_f1={f1_text}, discard_f2={f2_text}, "
        f"valid={valid_text})"
    )


def plot_metric(
    points_by_model: dict[str, list[ResultPoint]],
    dataset: str,
    metric_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    if not points_by_model:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    ax_base, ax_source = axes

    for model_name in sorted(points_by_model):
        points = points_by_model[model_name]
        x = [p.nua for p in points]
        y_base = [p.base_acc for p in points]
        y_source = [p.source_acc for p in points]
        label = make_legend_label(model_name, points, metric_name)

        ax_base.plot(x, y_base, marker="o", linewidth=1.8, label=label)
        ax_source.plot(x, y_source, marker="o", linewidth=1.8, label=label)

    ax_base.set_title(f"{dataset} | {metric_name} | Base")
    ax_source.set_title(f"{dataset} | {metric_name} | Source")
    for ax in axes:
        ax.set_xlabel("Num Attractors")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)

    handles, labels = ax_base.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=1,
        frameon=True,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    observations_dir = Path(args.observations_dir).resolve()
    if not observations_dir.exists():
        raise SystemExit(f"Observations directory not found: {observations_dir}")

    plots_root = observations_dir / args.plots_subdir / models_to_use
    generated = 0

    for dataset in args.datasets:
        main_points, aux_points = collect_dataset_points(observations_dir, dataset)
        dataset_plot_dir = plots_root / dataset

        main_out = dataset_plot_dir / "main_verb_accuracy.png"
        aux_out = dataset_plot_dir / "aux_verb_accuracy.png"

        if main_points:
            plot_metric(
                points_by_model=main_points,
                dataset=dataset,
                metric_name="Main Verb",
                output_path=main_out,
                dpi=args.dpi,
            )
            generated += 1

        if aux_points:
            plot_metric(
                points_by_model=aux_points,
                dataset=dataset,
                metric_name="Aux Verb",
                output_path=aux_out,
                dpi=args.dpi,
            )
            generated += 1

    if generated == 0:
        raise SystemExit("No plots generated. Check observations directory layout.")

    print(f"Saved {generated} plot(s) under: {plots_root}")


if __name__ == "__main__":
    main()
