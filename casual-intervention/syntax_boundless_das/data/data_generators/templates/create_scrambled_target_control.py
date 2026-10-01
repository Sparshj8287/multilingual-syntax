#!/usr/bin/env python3
"""Create a scrambled-target control dataset for obj_rel_across_anim linear_4."""

import argparse
import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

SOURCE_DATASET = "obj_rel_across_anim"
TARGET_DATASET = "corrupted_obj_rel_across_anim"
HYPOTHESIS = "linear"
NUM_ATTRACTORS = 4
SOURCE_FIELDS = ("source_sentence", "MS_source", "MV_source")


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def source_bundle(row):
    return {field: row[field] for field in SOURCE_FIELDS}


def shuffled_source_bundles(rows):
    bundles = [source_bundle(row) for row in rows]
    if len(bundles) < 2:
        raise ValueError("Need at least two rows to scramble source fields.")

    shuffled = bundles[:]
    random.shuffle(shuffled)

    # Avoid leaving any row paired with its original source bundle when possible.
    attempts = 1
    while any(original == new for original, new in zip(bundles, shuffled)):
        if attempts >= 1000:
            raise RuntimeError("Could not produce a fully scrambled source assignment.")
        random.shuffle(shuffled)
        attempts += 1

    return shuffled


def scramble_rows(rows):
    shuffled_sources = shuffled_source_bundles(rows)
    corrupted_rows = []

    for row, shuffled_source in zip(rows, shuffled_sources):
        corrupted_row = row.copy()
        for field in SOURCE_FIELDS:
            corrupted_row[field] = shuffled_source[field]
        corrupted_rows.append(corrupted_row)

    return corrupted_rows


def build_paths(split):
    input_name = f"{SOURCE_DATASET}_pairs_{HYPOTHESIS}_{NUM_ATTRACTORS}.jsonl"
    output_name = f"{TARGET_DATASET}_pairs_{HYPOTHESIS}_{NUM_ATTRACTORS}.jsonl"

    input_path = DATA_DIR / SOURCE_DATASET / split / HYPOTHESIS / input_name
    output_path = DATA_DIR / TARGET_DATASET / split / HYPOTHESIS / output_name
    return input_path, output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle source_sentence, MS_source, and MV_source across "
            "obj_rel_across_anim linear_4 rows while preserving base fields."
        )
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test"),
        help="Dataset split to corrupt. Defaults to the requested train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random.shuffle output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    input_path, output_path = build_paths(args.split)
    rows = load_jsonl(input_path)
    corrupted_rows = scramble_rows(rows)
    write_jsonl(output_path, corrupted_rows)

    print(f"Read {len(rows)} rows from {input_path}")
    print(f"Wrote scrambled-target control rows to {output_path}")


if __name__ == "__main__":
    main()
