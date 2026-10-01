import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


LAYER_RE = re.compile(r"^layer-(\d+)$")
ATTRACTOR_RE = re.compile(r"^attractors_(\d+)$")


@dataclass(frozen=True)
class MetricPoint:
    dataset_name: str
    variation: str
    sentence_type: str
    attractor_count: int
    model_name: str
    layer: int
    task_name: str
    metric_file: str
    metric_path: Path
    modified_time: float


def _parse_metric_path(metric_path: Path, experiments_dir: Path) -> MetricPoint | None:
    """Parse corpus-layout experiment paths into plotting metadata."""
    try:
        rel_parts = metric_path.relative_to(experiments_dir).parts
    except ValueError:
        return None

    # Expected suffix:
    # dataset/variation/sentence_type/attractors_N/model/layer-X/task/run/metric
    # The experiments root may also include a namespace prefix, such as
    # syntax_boundless/dataset/variation/...
    if len(rel_parts) < 9:
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
    attractor_dir = rel_parts[attractor_index]
    model_name = rel_parts[attractor_index + 1]
    layer_dir = rel_parts[attractor_index + 2]
    task_name = rel_parts[attractor_index + 3]
    metric_file = rel_parts[-1]

    attractor_match = ATTRACTOR_RE.match(attractor_dir)
    layer_match = LAYER_RE.match(layer_dir)
    if attractor_match is None or layer_match is None:
        return None

    try:
        value = float(metric_path.read_text().strip())
    except (OSError, ValueError):
        return None

    if not value == value:  # skip NaN without importing math
        return None

    return MetricPoint(
        dataset_name=dataset_name,
        variation=variation,
        sentence_type=sentence_type,
        attractor_count=int(attractor_match.group(1)),
        model_name=model_name,
        layer=int(layer_match.group(1)),
        task_name=task_name,
        metric_file=metric_file,
        metric_path=metric_path,
        modified_time=metric_path.stat().st_mtime,
    )


def _read_metric_value(metric_path: Path) -> float:
    return float(metric_path.read_text().strip())


def _discover_points(experiments_dir: Path, metric_file: str) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for metric_path in experiments_dir.rglob(metric_file):
        point = _parse_metric_path(metric_path, experiments_dir)
        if point is not None:
            points.append(point)
    return points


def _filter_points(points: list[MetricPoint], args: argparse.Namespace) -> list[MetricPoint]:
    def matches(point: MetricPoint) -> bool:
        filters = {
            "dataset_name": args.dataset_name,
            "variation": args.variation,
            "sentence_type": args.sentence_type,
            "model_name": args.model,
            "task_name": args.task_name,
        }
        for attr, expected in filters.items():
            if expected and getattr(point, attr) != expected:
                return False
        if args.attractor_counts and point.attractor_count not in args.attractor_counts:
            return False
        return True

    return [point for point in points if matches(point)]


def _keep_latest_per_layer(points: list[MetricPoint]) -> list[MetricPoint]:
    latest: dict[tuple[str, str, str, int, str, int, str, str], MetricPoint] = {}
    for point in points:
        key = (
            point.dataset_name,
            point.variation,
            point.sentence_type,
            point.attractor_count,
            point.model_name,
            point.layer,
            point.task_name,
            point.metric_file,
        )
        previous = latest.get(key)
        if previous is None or point.modified_time > previous.modified_time:
            latest[key] = point
    return list(latest.values())


def _format_title(dataset_name: str, variation: str) -> str:
    return f"{dataset_name} ({variation.capitalize()})"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _plot_group(
    group_key: tuple[str, str, str, str, str, str],
    points: list[MetricPoint],
    plots_dir: Path,
    dpi: int,
) -> Path:
    dataset_name, variation, sentence_type, model_name, task_name, metric_file = group_key
    by_attractor: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for point in points:
        by_attractor[point.attractor_count].append(
            (point.layer, _read_metric_value(point.metric_path))
        )

    styles = [
        ("o", "-", "#1f77b4"),
        ("s", "--", "#2ca02c"),
        ("^", ":", "#d62728"),
        ("D", "-.", "#9467bd"),
        ("v", "-", "#ff7f0e"),
        ("P", "--", "#17becf"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, attractor_count in enumerate(sorted(by_attractor)):
        series = sorted(by_attractor[attractor_count])
        layers = [layer for layer, _ in series]
        values = [value for _, value in series]
        marker, linestyle, color = styles[index % len(styles)]
        ax.plot(
            layers,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            markersize=7,
            label=f"attractors_{attractor_count}",
        )

    ax.set_title(_format_title(dataset_name, variation), fontsize=18, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel(metric_file, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    all_layers = sorted({point.layer for point in points})
    if all_layers:
        ax.set_xticks(all_layers)

    output_dir = plots_dir / dataset_name / variation / sentence_type / model_name / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(metric_file)}_by_attractors.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot layer-wise probing curves grouped by attractor count."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments",
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "visualizations_plots" / "layer_curves",
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--metric-file",
        default="test.uuas",
        help="Metric file to plot, for example test.uuas or dev.uuas.",
    )
    parser.add_argument("--dataset-name", help="Only plot this dataset name.")
    parser.add_argument("--variation", help="Only plot this variation, e.g. linear or bow.")
    parser.add_argument("--sentence-type", help="Only plot this sentence type.")
    parser.add_argument("--model", help="Only plot this model directory name.")
    parser.add_argument("--task-name", help="Only plot this probing task.")
    parser.add_argument(
        "--attractor-counts",
        type=int,
        nargs="+",
        help="Only include these attractor counts.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    args = parser.parse_args()

    points = _discover_points(args.experiments_dir, args.metric_file)
    points = _filter_points(points, args)
    points = _keep_latest_per_layer(points)

    if not points:
        print(f"No {args.metric_file} values found under {args.experiments_dir}")
        return

    groups: dict[tuple[str, str, str, str, str, str], list[MetricPoint]] = defaultdict(list)
    for point in points:
        group_key = (
            point.dataset_name,
            point.variation,
            point.sentence_type,
            point.model_name,
            point.task_name,
            point.metric_file,
        )
        groups[group_key].append(point)

    for group_key, group_points in sorted(groups.items()):
        output_path = _plot_group(group_key, group_points, args.plots_dir, args.dpi)
        rel_output = os.path.relpath(output_path, Path.cwd())
        print(f"Saved {rel_output}")


if __name__ == "__main__":
    main()
