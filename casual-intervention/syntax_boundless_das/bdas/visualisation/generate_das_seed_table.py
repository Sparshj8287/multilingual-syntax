#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


VIS_DIR = Path(__file__).resolve().parent
BDAS_ROOT = VIS_DIR.parent
DEFAULT_SEED_42_RESULTS_DIR = BDAS_ROOT / "results_das_full"
DEFAULT_SEED_42_FALLBACK_RESULTS_DIR = BDAS_ROOT / "results_das_full_es"
DEFAULT_SEED_SWEEP_RESULTS_DIR = BDAS_ROOT / "results_das_full_with_saved_boundaries"
DEFAULT_OUTPUT_DIR = VIS_DIR / "tables_das_seeds"

ATTRACTOR = 4
BASELINE_SEED = "seed_42"
SWEEP_SEEDS = ("seed_96", "seed_1120")
ALL_SEEDS = (BASELINE_SEED, *SWEEP_SEEDS)

DIRECTIONS = {
    "source_to_base": "SG -> PL",
    "base_to_source": "PL -> SG",
}
HYPOTHESES = {
    "syn": {
        "label": "H_syn (Syntax)",
        "inputs": [("obj_rel_across_anim", "linear"), ("obj_rel_across_anim", "bow")],
    },
    "prox": {
        "label": "H_prox (Proximity)",
        "inputs": [("obj_rel_across_anim_2", "linear")],
    },
    "maj": {
        "label": "H_maj (Majority)",
        "inputs": [("obj_rel_across_anim_2", "bow")],
    },
}
MODEL_LABELS = {
    "gemma-3-12b-pt": "Gemma-3-12B",
    "qwen-3-8b": "Qwen-3-8B",
    "olmo-3-7b": "Olmo-3-7B",
    "llama-3.1-8b": "Llama-3.1-8B",
}
MODEL_ALIASES = {
    "gemma": "gemma-3-12b-pt",
    "qwen": "qwen-3-8b",
    "olmo": "olmo-3-7b",
    "llama": "llama-3.1-8b",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-layer mean +/- standard deviation table of Boundless DAS IIA "
            f"across seeds {', '.join(ALL_SEEDS)} for the {ATTRACTOR}-attractor case."
        )
    )
    parser.add_argument(
        "--seed-42-results-dir",
        type=Path,
        default=DEFAULT_SEED_42_RESULTS_DIR,
        help=f"Directory holding unprefixed seed-42 results. Default: {DEFAULT_SEED_42_RESULTS_DIR}",
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
        "--seed-sweep-results-dir",
        type=Path,
        default=DEFAULT_SEED_SWEEP_RESULTS_DIR,
        help=(
            "Directory holding seed_96 and seed_1120 results. "
            f"Default: {DEFAULT_SEED_SWEEP_RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the CSV and JSON tables. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model aliases to include, e.g. --models gemma qwen. Default: all.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Report IIA as percentages instead of decimals.",
    )
    return parser.parse_args()


def normalize_model_filters(raw_models):
    models = []
    for raw_model in raw_models:
        models.extend(part.strip().lower() for part in raw_model.split(","))
    models = [model for model in models if model]
    if not models or "all" in models:
        return None
    return {MODEL_ALIASES.get(model, model) for model in models}


def model_sort_key(model):
    order = {"gemma-3-12b-pt": 0, "qwen-3-8b": 1, "olmo-3-7b": 2, "llama-3.1-8b": 3}
    return (order.get(model, 99), model)


def read_seed_results(source_dir, wanted_seeds, model_filters):
    """Map (model, direction, dataset, variation, seed) -> {layer: accuracy}."""
    results = {}
    for summary_path in source_dir.rglob("all_layers_summary.json"):
        parts = summary_path.relative_to(source_dir).parts
        if len(parts) < 6:
            continue

        model, dataset, variation = parts[0], parts[1], parts[2]
        seed = parts[3] if parts[3].startswith("seed_") else BASELINE_SEED
        if seed not in wanted_seeds:
            continue
        if model_filters is not None and model not in model_filters:
            continue

        attractor_index = 4 if seed != BASELINE_SEED else 3
        if len(parts) <= attractor_index + 2:
            continue
        try:
            attractor = int(parts[attractor_index].split("_", 1)[0])
        except ValueError:
            continue
        if attractor != ATTRACTOR:
            continue

        direction = parts[attractor_index + 1]
        if direction not in DIRECTIONS:
            continue

        summary = json.loads(summary_path.read_text())
        series = results.setdefault((model, direction, dataset, variation, seed), {})
        for layer_info in summary.get("layers", []):
            layer = layer_info.get("layer")
            accuracy = layer_info.get("final_test_accuracy")
            if layer is not None and accuracy is not None:
                series[int(layer)] = float(accuracy)
    return results


def hypothesis_value(results, model, direction, hypothesis, seed, layer):
    """Average the hypothesis' input runs for one seed and layer, or None if incomplete."""
    values = []
    for dataset, variation in HYPOTHESES[hypothesis]["inputs"]:
        series = results.get((model, direction, dataset, variation, seed))
        if series is None or layer not in series:
            return None
        values.append(series[layer])
    return statistics.fmean(values)


def build_rows(results, model_filters):
    models = sorted({key[0] for key in results}, key=model_sort_key)
    if model_filters is not None:
        models = [model for model in models if model in model_filters]

    rows = []
    for model in models:
        for direction in DIRECTIONS:
            layers = sorted(
                {
                    layer
                    for key, series in results.items()
                    if key[0] == model and key[1] == direction
                    for layer in series
                }
            )
            for hypothesis in HYPOTHESES:
                for layer in layers:
                    seed_values = {
                        seed: value
                        for seed in ALL_SEEDS
                        for value in [
                            hypothesis_value(
                                results, model, direction, hypothesis, seed, layer
                            )
                        ]
                        if value is not None
                    }
                    if not seed_values:
                        continue
                    values = list(seed_values.values())
                    rows.append(
                        {
                            "model": model,
                            "model_label": MODEL_LABELS.get(model, model),
                            "direction": direction,
                            "setup": DIRECTIONS[direction],
                            "hypothesis": hypothesis,
                            "hypothesis_label": HYPOTHESES[hypothesis]["label"],
                            "layer": layer,
                            "n_seeds": len(values),
                            "seeds": ",".join(sorted(seed_values, key=ALL_SEEDS.index)),
                            "mean": statistics.fmean(values),
                            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                            **{
                                f"{seed}": seed_values.get(seed)
                                for seed in ALL_SEEDS
                            },
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

    csv_path = output_dir / f"das_iia_seed_mean_std_{suffix}.csv"
    fieldnames = [
        "model",
        "model_label",
        "direction",
        "setup",
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

    json_path = output_dir / f"das_iia_seed_mean_std_{suffix}.json"
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path


def print_table(rows, percent):
    digits = 2 if percent else 4
    for model in dict.fromkeys(row["model"] for row in rows):
        for direction in DIRECTIONS:
            panel = [
                row
                for row in rows
                if row["model"] == model and row["direction"] == direction
            ]
            if not panel:
                continue
            print(f"\n{MODEL_LABELS.get(model, model)} ({DIRECTIONS[direction]})")
            layers = sorted({row["layer"] for row in panel})
            header = f"{'Layer':>6}" + "".join(
                f"{HYPOTHESES[hypothesis]['label']:>28}" for hypothesis in HYPOTHESES
            )
            print(header)
            for layer in layers:
                cells = []
                for hypothesis in HYPOTHESES:
                    match = next(
                        (
                            row
                            for row in panel
                            if row["layer"] == layer and row["hypothesis"] == hypothesis
                        ),
                        None,
                    )
                    if match is None:
                        cells.append(f"{'-':>28}")
                    else:
                        marker = "" if match["n_seeds"] == len(ALL_SEEDS) else f" (n={match['n_seeds']})"
                        cells.append(
                            f"{match['mean']:.{digits}f} ± {match['std']:.{digits}f}{marker}".rjust(28)
                        )
                print(f"{layer:>6}" + "".join(cells))


def main():
    args = parse_args()
    model_filters = normalize_model_filters(args.models)

    # Read the fallback first so primary seed-42 results overwrite it.
    results = read_seed_results(
        args.seed_42_fallback_results_dir.expanduser().resolve(),
        {BASELINE_SEED},
        model_filters,
    )
    results.update(
        read_seed_results(
            args.seed_42_results_dir.expanduser().resolve(),
            {BASELINE_SEED},
            model_filters,
        )
    )
    results.update(
        read_seed_results(
            args.seed_sweep_results_dir.expanduser().resolve(),
            set(SWEEP_SEEDS),
            model_filters,
        )
    )
    if not results:
        print("No attractor-4 results found.")
        return

    rows = scale_rows(build_rows(results, model_filters), 100 if args.percent else 1)
    if not rows:
        print("No hypothesis series could be assembled.")
        return

    print_table(rows, args.percent)

    incomplete = [row for row in rows if row["n_seeds"] < len(ALL_SEEDS)]
    if incomplete:
        missing = sorted(
            {
                (
                    row["model_label"],
                    row["setup"],
                    row["hypothesis_label"],
                    row["seeds"],
                )
                for row in incomplete
            }
        )
        print("\n[warn] cells computed from fewer than three seeds:")
        for model_label, setup, hypothesis_label, seeds in missing:
            print(f"  - {model_label} ({setup}) {hypothesis_label}: {seeds}")

    csv_path, json_path = write_outputs(rows, args.output_dir.expanduser().resolve(), args.percent)
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
