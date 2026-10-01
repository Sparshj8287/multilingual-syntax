#!/usr/bin/env python3
"""Aggregate MAR across seeds into mean +/- standard deviation for every model.

Seed 42 lives in the unprefixed result directories and was only trained at one layer
per model, sometimes with the syntax and heuristic runs sitting at different layers.
Seeds 96 and 1120 have full layer sweeps. To keep the three seeds comparable, the
default `anchor` mode reuses the exact (syntax_layer, heuristic_layer) pair that seed
42 used and evaluates every seed at that same pair. `matched` mode instead walks every
layer where both sides of a comparison exist at the same layer.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from matplotlib import font_manager
import matplotlib.pyplot as plt

from compute_mar_seed_layers import (
    DIRECTIONS,
    MODEL_LABELS,
    SCRIPT_DIR,
    BDAS_ROOT,
    boundary_from_result,
    load_basis,
    load_json,
    mar_from_bases,
    model_sort_key,
    normalize_model_filters,
    normalize_model_name,
    result_checkpoint_path,
)


DEFAULT_RESULTS_ROOT = BDAS_ROOT / "results_das_full_with_saved_boundaries"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_seed_stats"

BASELINE_SEED = 42
DEFAULT_SEEDS = (BASELINE_SEED, 96, 1120)

COMPARISONS = {
    "syntax_linear_vs_prox": {
        "label": r"$\mathcal{H}_{prox}$",
        "syntax": ("obj_rel_across_anim", "linear"),
        "heuristic": ("obj_rel_across_anim_2", "linear"),
        "color": "#E64B35",
    },
    "syntax_bow_vs_maj": {
        "label": r"$\mathcal{H}_{maj}$",
        "syntax": ("obj_rel_across_anim", "bow"),
        "heuristic": ("obj_rel_across_anim_2", "bow"),
        "color": "#3C5488",
    },
}
INCOMPLETE_MARKER = r"$^{\dagger}$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model aliases to include. Default: all.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Seeds to aggregate. Default: {' '.join(str(s) for s in DEFAULT_SEEDS)}.",
    )
    parser.add_argument("--nua", type=int, default=4, help="Attractor count. Default: 4.")
    parser.add_argument(
        "--layer-mode",
        choices=["anchor", "matched"],
        default="anchor",
        help=(
            "anchor: evaluate every seed at the layer pair seed 42 used. "
            "matched: evaluate every layer where both sides exist at the same layer. "
            "Default: anchor."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which comparisons would be computed without loading checkpoints.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help=(
            "Re-aggregate and re-render from an existing mar_per_seed.csv instead of "
            "reloading rotation checkpoints."
        ),
    )
    return parser.parse_args()


def configure_table_style() -> None:
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
        {"font.family": font_family, "font.size": 11, "mathtext.fontset": "stix"}
    )


def seed_from_path(result_path: Path, results_root: Path) -> int:
    for part in result_path.relative_to(results_root).parts:
        if part.startswith("seed_"):
            return int(part.removeprefix("seed_"))
    return BASELINE_SEED


def index_results(results_root: Path, nua: int, models, seeds):
    """Map (seed, model, direction, dataset, variation, layer) -> entry."""
    indexed = {}
    for result_path in results_root.rglob("result.json"):
        try:
            seed = seed_from_path(result_path, results_root)
        except ValueError:
            continue
        if seed not in seeds:
            continue

        payload = load_json(result_path)
        model = normalize_model_name(str(payload["model"]["output_dir_name"]))
        if models is not None and model not in models:
            continue
        if int(payload["dataset"]["nua"]) != nua:
            continue

        try:
            boundary = boundary_from_result(payload)
            checkpoint_path = result_checkpoint_path(payload)
        except ValueError:
            continue
        if not checkpoint_path.exists():
            continue

        key = (
            seed,
            model,
            str(payload["intervention"]["direction"]),
            str(payload["dataset"]["dataset_name"]),
            str(payload["dataset"]["variation"]),
            int(payload["intervention"]["layer"]),
        )
        indexed[key] = {
            "result_path": result_path,
            "boundary": boundary,
            "checkpoint_path": checkpoint_path,
        }
    return indexed


def layers_for(indexed, seed, model, direction, source):
    dataset, variation = source
    return sorted(
        key[5]
        for key in indexed
        if key[0] == seed
        and key[1] == model
        and key[2] == direction
        and key[3] == dataset
        and key[4] == variation
    )


def plan_comparisons(indexed, seeds, layer_mode):
    """Yield (model, direction, comparison, syntax_layer, heuristic_layer, seed) tasks."""
    models = sorted({key[1] for key in indexed}, key=model_sort_key)
    plans = []
    warnings = []

    for model in models:
        for direction in DIRECTIONS:
            for comparison, spec in COMPARISONS.items():
                if layer_mode == "anchor":
                    syntax_layers = layers_for(
                        indexed, BASELINE_SEED, model, direction, spec["syntax"]
                    )
                    heuristic_layers = layers_for(
                        indexed, BASELINE_SEED, model, direction, spec["heuristic"]
                    )
                    if not syntax_layers or not heuristic_layers:
                        warnings.append(
                            f"{model}/{direction}/{comparison}: no seed-{BASELINE_SEED} "
                            "run to anchor the layer pair"
                        )
                        continue
                    layer_pairs = [(max(syntax_layers), max(heuristic_layers))]
                else:
                    shared = set.intersection(
                        *(
                            set(layers_for(indexed, seed, model, direction, source))
                            for seed in seeds
                            for source in (spec["syntax"], spec["heuristic"])
                        )
                        or [set()]
                    )
                    layer_pairs = [(layer, layer) for layer in sorted(shared)]
                    if not layer_pairs:
                        warnings.append(
                            f"{model}/{direction}/{comparison}: no layer shared by all seeds"
                        )
                        continue

                for syntax_layer, heuristic_layer in layer_pairs:
                    plans.append(
                        {
                            "model": model,
                            "direction": direction,
                            "comparison": comparison,
                            "syntax_layer": syntax_layer,
                            "heuristic_layer": heuristic_layer,
                        }
                    )
    return plans, warnings


def compute_seed_rows(indexed, plans, seeds, dry_run):
    seed_rows = []
    missing = []
    total = len(plans) * len(seeds)
    done = 0

    for plan in plans:
        spec = COMPARISONS[plan["comparison"]]
        for seed in seeds:
            done += 1
            syntax_key = (
                seed,
                plan["model"],
                plan["direction"],
                *spec["syntax"],
                plan["syntax_layer"],
            )
            heuristic_key = (
                seed,
                plan["model"],
                plan["direction"],
                *spec["heuristic"],
                plan["heuristic_layer"],
            )
            syntax_entry = indexed.get(syntax_key)
            heuristic_entry = indexed.get(heuristic_key)
            if syntax_entry is None or heuristic_entry is None:
                missing.append(
                    f"seed_{seed}/{plan['model']}/{plan['direction']}/{plan['comparison']} "
                    f"@ layers {plan['syntax_layer']}/{plan['heuristic_layer']}"
                )
                continue

            print(
                f"[{done}/{total}] seed_{seed} {plan['model']} {plan['direction']} "
                f"{plan['comparison']} layers {plan['syntax_layer']}/{plan['heuristic_layer']}",
                flush=True,
            )
            if dry_run:
                continue

            W_syntax = load_basis(syntax_entry)
            W_heuristic = load_basis(heuristic_entry)
            seed_rows.append(
                {
                    "seed": seed,
                    "model": plan["model"],
                    "model_label": MODEL_LABELS.get(plan["model"], plan["model"]),
                    "direction": plan["direction"],
                    "setup": DIRECTIONS[plan["direction"]]["label"],
                    "comparison": plan["comparison"],
                    "syntax_layer": plan["syntax_layer"],
                    "heuristic_layer": plan["heuristic_layer"],
                    "mar": mar_from_bases(W_syntax, W_heuristic),
                    "syntax_m": int(W_syntax.shape[1]),
                    "heuristic_m": int(W_heuristic.shape[1]),
                }
            )
            del W_syntax, W_heuristic
    return seed_rows, missing


def aggregate(seed_rows, seeds):
    grouped = defaultdict(dict)
    for row in seed_rows:
        key = (
            row["model"],
            row["direction"],
            row["comparison"],
            row["syntax_layer"],
            row["heuristic_layer"],
        )
        grouped[key][row["seed"]] = row

    comparison_order = list(COMPARISONS)
    rows = []
    for key in sorted(
        grouped,
        key=lambda k: (model_sort_key(k[0]), k[1], comparison_order.index(k[2]), k[3]),
    ):
        model, direction, comparison, syntax_layer, heuristic_layer = key
        by_seed = grouped[key]
        values = [by_seed[seed]["mar"] for seed in seeds if seed in by_seed]
        if not values:
            continue
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "direction": direction,
                "setup": DIRECTIONS[direction]["label"],
                "comparison": comparison,
                "syntax_layer": syntax_layer,
                "heuristic_layer": heuristic_layer,
                "n_seeds": len(values),
                "seeds": ",".join(str(seed) for seed in seeds if seed in by_seed),
                "mar_mean": statistics.fmean(values),
                "mar_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                **{f"mar_seed_{seed}": by_seed.get(seed, {}).get("mar") for seed in seeds},
                **{f"syntax_m_seed_{seed}": by_seed.get(seed, {}).get("syntax_m") for seed in seeds},
                **{
                    f"heuristic_m_seed_{seed}": by_seed.get(seed, {}).get("heuristic_m")
                    for seed in seeds
                },
            }
        )
    return rows


def write_outputs(seed_rows, summary_rows, output_dir, seeds, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = output_dir / "mar_per_seed.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    print(f"[saved] {per_seed_csv}")

    summary_csv = output_dir / "mar_seed_mean_std.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[saved] {summary_csv}")

    summary_json = output_dir / "mar_seed_mean_std.json"
    summary_json.write_text(
        json.dumps(
            {**metadata, "seeds": seeds, "per_seed": seed_rows, "summary": summary_rows},
            indent=2,
        )
    )
    print(f"[saved] {summary_json}")
    return summary_json


def print_table(summary_rows, seeds):
    header = (
        f"{'Model':16s}{'Direction':12s}{'Comparison':22s}"
        f"{'Layers':10s}{'MAR mean ± std':>22s}{'seeds':>8s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary_rows:
        layers = f"{row['syntax_layer']}/{row['heuristic_layer']}"
        value = f"{row['mar_mean']:.4f} ± {row['mar_std']:.4f}"
        print(
            f"{row['model_label']:16s}{row['setup']:12s}{row['comparison']:22s}"
            f"{layers:10s}{value:>22s}{row['n_seeds']:>8d}"
        )


def render_table(summary_rows, output_dir, seed_count):
    configure_table_style()
    columns = [
        "Model",
        "Direction",
        "Comparison",
        "Layer (syn/heur)",
        r"MAR Mean $\pm$ Std",
    ]
    cell_text = []
    for row in summary_rows:
        marker = INCOMPLETE_MARKER if row["n_seeds"] < seed_count else ""
        cell_text.append(
            [
                row["model_label"],
                DIRECTIONS[row["direction"]]["setup"],
                COMPARISONS[row["comparison"]]["label"],
                f"{row['syntax_layer']} / {row['heuristic_layer']}",
                rf"${row['mar_mean']:.4f}\,\pm\,{row['mar_std']:.4f}${marker}",
            ]
        )

    has_marker = any(INCOMPLETE_MARKER in cell for row in cell_text for cell in row)
    figure_height = 0.34 * (len(cell_text) + 1) + (0.6 if has_marker else 0.35)
    fig, ax = plt.subplots(figsize=(8.2, figure_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        colLoc="center",
        cellLoc="left",
        colWidths=[0.2, 0.15, 0.16, 0.2, 0.26],
        bbox=[0.015, 0.02, 0.97, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

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
        elif row_idx == row_count:
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        else:
            cell.visible_edges = ""
        if col_idx >= 2:
            cell.get_text().set_ha("center")
        if row_idx > 0 and col_idx == 2:
            cell.get_text().set_color(
                COMPARISONS[summary_rows[row_idx - 1]["comparison"]]["color"]
            )

    for row_idx in range(2, row_count + 1):
        if summary_rows[row_idx - 1]["model"] != summary_rows[row_idx - 2]["model"]:
            for col_idx in range(len(columns)):
                cell = table[(row_idx, col_idx)]
                cell.visible_edges = "T" + ("B" if row_idx == row_count else "")
                cell.set_linewidth(0.6)

    if has_marker:
        fig.text(
            0.015,
            0.004,
            rf"$^{{\dagger}}$Computed from fewer than {seed_count} seeds.",
            fontsize=9,
            ha="left",
            va="bottom",
        )

    for suffix in ("pdf", "png"):
        output_path = output_dir / f"mar_seed_mean_std_table.{suffix}"
        fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.04)
        print(f"[saved] {output_path}")
    plt.close(fig)


def read_per_seed_csv(path: Path):
    integer_fields = {"seed", "syntax_layer", "heuristic_layer", "syntax_m", "heuristic_m"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            typed = dict(row)
            for field in integer_fields:
                typed[field] = int(typed[field])
            typed["mar"] = float(typed["mar"])
            rows.append(typed)
    return rows


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    seeds = list(dict.fromkeys(args.seeds))
    model_filters = normalize_model_filters(args.models)
    output_dir = args.output_dir.expanduser().resolve()

    if args.reuse:
        per_seed_csv = output_dir / "mar_per_seed.csv"
        if not per_seed_csv.exists():
            raise SystemExit(f"--reuse needs an existing {per_seed_csv}")
        seed_rows = read_per_seed_csv(per_seed_csv)
        summary_rows = aggregate(seed_rows, seeds)
        print_table(summary_rows, seeds)
        write_outputs(
            seed_rows,
            summary_rows,
            output_dir,
            seeds,
            {
                "results_root": str(results_root),
                "nua": args.nua,
                "layer_mode": args.layer_mode,
            },
        )
        render_table(summary_rows, output_dir, len(seeds))
        return

    indexed = index_results(results_root, args.nua, model_filters, set(seeds))
    if not indexed:
        raise SystemExit(f"No usable result.json files found in {results_root}")
    print(f"[index] {len(indexed)} runs with boundaries and checkpoints")

    plans, warnings = plan_comparisons(indexed, seeds, args.layer_mode)
    if not plans:
        raise SystemExit("No comparisons could be planned.")
    print(f"[plan] {len(plans)} comparisons x {len(seeds)} seeds")

    seed_rows, missing = compute_seed_rows(indexed, plans, seeds, args.dry_run)
    for warning in warnings:
        print(f"[warn] {warning}")
    for item in missing:
        print(f"[warn] missing {item}")
    if args.dry_run:
        return
    if not seed_rows:
        raise SystemExit("No MAR values could be computed.")

    summary_rows = aggregate(seed_rows, seeds)
    print_table(summary_rows, seeds)

    write_outputs(
        seed_rows,
        summary_rows,
        output_dir,
        seeds,
        {
            "results_root": str(results_root),
            "nua": args.nua,
            "layer_mode": args.layer_mode,
        },
    )
    render_table(summary_rows, output_dir, len(seeds))


if __name__ == "__main__":
    main()
