"""
Prepare probing-ready train/dev/test splits from converted syntax_boundless_das CoNLL-U files.

Input layout (from convert_jsonl_to_conllu_stanza.py):
  datasets/syntax_boundless_das_conllu/<dataset>/<split>/<variation>/*.conllu

Output layout:
  datasets/syntax_boundless_das_ready/<dataset>/<variation>/<sentence_type>/attractors_<n>/
    - train.conllu
    - dev.conllu
    - test.conllu
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence, Tuple


ATTRACTOR_RE = re.compile(r"_(\d+)\.(base_sentence|source_sentence)\.conllu$")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    probing_root = script_path.parents[1]
    parser = argparse.ArgumentParser(
        description="Create train/dev/test CoNLL-U splits for syntax_boundless_das probing."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=probing_root / "datasets" / "syntax_boundless_das_conllu",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=probing_root / "datasets" / "syntax_boundless_das_ready",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=1000,
        help="Number of sentences removed from train and used as dev.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files.",
    )
    return parser.parse_args()


def read_conllu_sentences(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    blocks = []
    current = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current) + "\n\n")
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current) + "\n\n")
    return blocks


def write_sentences(path: Path, blocks: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for block in blocks:
            fout.write(block)


def parse_train_file_metadata(train_path: Path) -> Tuple[str, str]:
    match = ATTRACTOR_RE.search(train_path.name)
    if not match:
        raise ValueError(f"Could not parse attractor/sentence type from {train_path.name}")
    attractor_count, sentence_type = match.group(1), match.group(2)
    return attractor_count, sentence_type


def condition_output_dir(output_root: Path, dataset_name: str, variation: str, sentence_type: str, attractor_count: str) -> Path:
    return (
        output_root
        / dataset_name
        / variation
        / sentence_type
        / f"attractors_{attractor_count}"
    )


def main() -> None:
    args = parse_args()

    train_files = sorted(args.input_root.glob("*" + "/train/" + "*" + "/*.conllu"))
    if not train_files:
        raise FileNotFoundError(f"No train .conllu files found under {args.input_root}")

    prepared = 0
    skipped = 0
    for train_file in train_files:
        rel = train_file.relative_to(args.input_root)
        parts = rel.parts
        if len(parts) != 4:
            # Expected: <dataset>/train/<variation>/<file>.conllu
            continue
        dataset_name, split_name, variation, _ = parts
        if split_name != "train":
            continue

        attractor_count, sentence_type = parse_train_file_metadata(train_file)
        test_file = args.input_root / dataset_name / "test" / variation / train_file.name
        if not test_file.exists():
            print(f"[skip] Missing matching test file for {train_file}")
            skipped += 1
            continue

        out_dir = condition_output_dir(
            args.output_root, dataset_name, variation, sentence_type, attractor_count
        )
        train_out = out_dir / "train.conllu"
        dev_out = out_dir / "dev.conllu"
        test_out = out_dir / "test.conllu"

        if (
            not args.overwrite
            and train_out.exists()
            and dev_out.exists()
            and test_out.exists()
        ):
            print(f"[skip] {out_dir} already exists")
            skipped += 1
            continue

        train_blocks = read_conllu_sentences(train_file)
        if len(train_blocks) <= args.dev_size:
            raise ValueError(
                f"{train_file} has {len(train_blocks)} sentences; dev-size={args.dev_size} is too large."
            )
        dev_blocks = train_blocks[-args.dev_size :]
        new_train_blocks = train_blocks[: -args.dev_size]

        test_blocks = read_conllu_sentences(test_file)

        write_sentences(train_out, new_train_blocks)
        write_sentences(dev_out, dev_blocks)
        write_sentences(test_out, test_blocks)

        print(
            f"[done] {out_dir} | train={len(new_train_blocks)} dev={len(dev_blocks)} test={len(test_blocks)}"
        )
        prepared += 1

    print(f"Prepared {prepared} condition directories (skipped {skipped}).")


if __name__ == "__main__":
    main()
