"""
Convert JSONL sentence-pair datasets to CoNLL-U with Stanford Stanza.

This script is tailored for the syntax_boundless_das templates layout and
mirrors the input directory structure under an output dataset root.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import stanza

@dataclass
class SentenceRecord:
    sent_id: str
    text: str
    source_jsonl: str
    example_index: int
    field_name: str


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[2]
    probing_root = script_path.parents[1]

    default_input_root = (
        workspace_root
        / "casual-intervention"
        / "syntax_boundless_das"
        / "data"
        / "data_generators"
        / "templates"
        / "data"
    )
    default_output_root = probing_root / "datasets" / "syntax_boundless_das_conllu"

    parser = argparse.ArgumentParser(
        description=(
            "Convert syntax_boundless_das JSONL files to CoNLL-U files using "
            "Stanford Stanza dependency parses."
        )
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument(
        "--paradigms",
        nargs="+",
        default=["obj_rel_across_anim", "obj_rel_across_anim_2"],
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--variations", nargs="+", default=["linear", "bow"])
    parser.add_argument(
        "--sentence-fields",
        nargs="+",
        default=["base_sentence", "source_sentence"],
        help="JSON keys to read as sentences and convert to individual .conllu files.",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--processors", default="tokenize,pos,lemma,depparse")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap per input JSONL for debugging.",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download Stanza resources for the chosen language before running.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default, existing files are skipped.",
    )
    return parser.parse_args()


def discover_jsonl_files(
    input_root: Path,
    paradigms: Iterable[str],
    splits: Iterable[str],
    variations: Iterable[str],
) -> List[Path]:
    files: List[Path] = []
    for paradigm in paradigms:
        for split in splits:
            for variation in variations:
                search_dir = input_root / paradigm / split / variation
                if not search_dir.exists():
                    continue
                files.extend(sorted(search_dir.glob("*.jsonl")))
    return files


def conllu_block_from_sentence(record: SentenceRecord, sentence, sentence_idx: int) -> str:
    if sentence_idx == 1:
        sent_id = record.sent_id
    else:
        sent_id = f"{record.sent_id}.{sentence_idx}"

    text = sentence.text or record.text
    text = text.replace("\n", " ").strip()

    lines = [
        f"# sent_id = {sent_id}",
        f"# text = {text}",
        f"# source_jsonl = {record.source_jsonl}",
        f"# example_index = {record.example_index}",
        f"# sentence_field = {record.field_name}",
    ]

    for idx, word in enumerate(sentence.words, start=1):
        form = (word.text or "_").replace("\t", " ")
        lemma = (word.lemma or "_").replace("\t", " ")
        upos = word.upos or "_"
        xpos = word.xpos or "_"
        feats = word.feats or "_"
        head = word.head if word.head is not None else 0
        deprel = word.deprel or "_"
        deps = "_"
        misc = "_"
        lines.append(
            f"{idx}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}"
        )

    return "\n".join(lines)


def write_batch(nlp, records: List[SentenceRecord], handle) -> None:
    docs = nlp.bulk_process([r.text for r in records])
    if len(docs) != len(records):
        raise RuntimeError(
            f"Stanza returned {len(docs)} docs for {len(records)} input sentences."
        )

    for record, doc in zip(records, docs):
        if not doc.sentences:
            continue
        for sentence_idx, sentence in enumerate(doc.sentences, start=1):
            block = conllu_block_from_sentence(record, sentence, sentence_idx)
            handle.write(block)
            handle.write("\n\n")


def convert_field(
    jsonl_path: Path,
    field_name: str,
    input_root: Path,
    output_root: Path,
    nlp,
    batch_size: int,
    overwrite: bool,
    max_examples: int | None,
) -> int:
    relative = jsonl_path.relative_to(input_root)
    output_dir = output_root / relative.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{jsonl_path.stem}.{field_name}.conllu"

    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists (use --overwrite to regenerate)")
        return 0

    converted = 0
    batch: List[SentenceRecord] = []
    source_jsonl_rel = str(relative)

    with jsonl_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_idx, line in enumerate(fin, start=1):
            if max_examples is not None and line_idx > max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(field_name)
            if not text or not isinstance(text, str):
                continue

            batch.append(
                SentenceRecord(
                    sent_id=f"{jsonl_path.stem}:{line_idx}:{field_name}",
                    text=text.strip(),
                    source_jsonl=source_jsonl_rel,
                    example_index=line_idx,
                    field_name=field_name,
                )
            )

            if len(batch) >= batch_size:
                write_batch(nlp, batch, fout)
                converted += len(batch)
                batch = []

        if batch:
            write_batch(nlp, batch, fout)
            converted += len(batch)

    print(f"[done] {output_path} ({converted} sentences)")
    return converted


def main() -> None:
    args = parse_args()



    if args.download_model:
        print(f"Downloading Stanza resources for language '{args.lang}'...")
        stanza.download(lang=args.lang, processors=args.processors, verbose=True)

    print("Initializing Stanza pipeline...")
    nlp = stanza.Pipeline(
        lang=args.lang,
        processors=args.processors,
        tokenize_pretokenized=False,
        verbose=False,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    jsonl_files = discover_jsonl_files(
        input_root=args.input_root,
        paradigms=args.paradigms,
        splits=args.splits,
        variations=args.variations,
    )
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found under {args.input_root} for the selected filters."
        )

    total = 0
    print(f"Found {len(jsonl_files)} JSONL files.")
    for jsonl_path in jsonl_files:
        print(f"Converting {jsonl_path} ...")
        for field_name in args.sentence_fields:
            total += convert_field(
                jsonl_path=jsonl_path,
                field_name=field_name,
                input_root=args.input_root,
                output_root=args.output_root,
                nlp=nlp,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                max_examples=args.max_examples,
            )

    print(f"Finished. Total converted sentences: {total}")


if __name__ == "__main__":
    main()
