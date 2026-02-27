#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Convert obj_rel_across_anim JSONL files into simple-agreement JSONL files "
            "by removing attractor clauses."
        )
    )
    parser.add_argument(
        "--input-root",
        default=str(script_dir / "data" / "obj_rel_across_anim"),
        help="Input root containing variation directories (default: data/obj_rel_across_anim).",
    )
    parser.add_argument(
        "--output-root",
        default=str(script_dir / "data" / "simple_agreement"),
        help="Output root for converted files (default: data/simple_agreement).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def to_simple_sentence(subject: str, verb: str) -> str:
    return f"The {subject.strip()} {verb.strip()}."


def transform_row(row: dict[str, Any]) -> dict[str, Any]:
    new_row = dict(row)

    ms_base = str(new_row.get("MS_base", "")).strip()
    mv_base = str(new_row.get("MV_base", "")).strip()
    ms_source = str(new_row.get("MS_source", "")).strip()
    mv_source = str(new_row.get("MV_source", "")).strip()

    if ms_base and mv_base:
        base_simple = to_simple_sentence(ms_base, mv_base)
        new_row["base_sentence"] = base_simple
        if "sentence_singular" in new_row:
            new_row["sentence_singular"] = base_simple

    if ms_source and mv_source:
        source_simple = to_simple_sentence(ms_source, mv_source)
        new_row["source_sentence"] = source_simple
        if "sentence_plural" in new_row:
            new_row["sentence_plural"] = source_simple

    # Requested: remove NUA in simple-agreement output.
    new_row.pop("NUA", None)
    return new_row


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    dup_count = 0
    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=True)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows, dup_count


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise ValueError(f"Input root does not exist: {input_root}")

    datasets_order = ["linear", "bow", "maximal"]
    dataset_dirs = [d for d in datasets_order if (input_root / d).exists()]

    # Include any extra variation dirs that may exist.
    for extra in sorted([p.name for p in input_root.iterdir() if p.is_dir()]):
        if extra not in dataset_dirs:
            dataset_dirs.append(extra)

    if not dataset_dirs:
        raise ValueError(f"No dataset directories found under: {input_root}")

    for dataset in dataset_dirs:
        dataset_dir = input_root / dataset
        jsonl_files = sorted(dataset_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"Skipping empty dataset directory: {dataset_dir}")
            continue

        print(f"\n=== Dataset: {dataset} ===")
        for jsonl_path in jsonl_files:
            rows = load_jsonl(jsonl_path)
            transformed = [transform_row(row) for row in rows]
            deduped, dup_count = deduplicate_rows(transformed)

            rel_path = jsonl_path.relative_to(input_root)
            out_path = output_root / rel_path
            write_jsonl(out_path, deduped)

            print(
                f"{rel_path}: input={len(rows)}, removed_duplicates={dup_count}, output={len(deduped)}"
            )

    print(f"\nSaved simple-agreement files to: {output_root}")


if __name__ == "__main__":
    main()
