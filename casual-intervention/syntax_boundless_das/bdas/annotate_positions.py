"""Annotate NUA=4 intervention pairs with token indices for p0 and p4.

p0 = last sub-token of the main subject head noun.
p4 = last sub-token of the final attractor head noun.

Indices are computed independently for the base and the source prefix, because
number inflection changes the tokenization (e.g. `consul` -> `cons` + `uls`), so
the two sides do not stay aligned. Indices are expressed in the same coordinate
system that `tokenize_prefix()` produces in the trainer, i.e. after any BOS token
has been prepended.

Originals are never modified: annotated copies are written to a parallel tree
under --out-root/<model_tag>/... so the running experiments keep reading the
untouched files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

# "<rel> the <noun>" -- the attractor frame used by every obj_rel_across_anim template.
# `[^\W\d_]` is "any Unicode letter": the lexicon contains accented nouns (café).
WORD = r"[^\W\d_]+"
ATTRACTOR_RE = re.compile(rf"\b(?:whom|that|who)\s+the\s+({WORD})")


def context_before_final_verb(sentence: str, final_verb: str) -> str:
    """Byte-for-byte copy of the trainer's prefix derivation."""
    text = sentence.strip()
    if text.endswith("."):
        text = text[:-1]
    suffix = f" {final_verb.strip()}"
    if text.endswith(suffix):
        return text[: -len(suffix)] + " "
    prefix, sep, _last = text.rpartition(" ")
    if not sep:
        raise ValueError(f"Cannot derive prefix from sentence: {sentence}")
    return prefix + " "


def subject_span(prefix: str, subject: str) -> tuple[int, int]:
    """Character span of the main subject head, which is the first noun token."""
    match = re.match(r"\s*(?:The|A|An|the|a|an)\s+", prefix)
    if not match:
        raise ValueError(f"Prefix does not start with a determiner: {prefix!r}")
    start = match.end()
    word = re.match(WORD, prefix[start:])
    if not word:
        raise ValueError(f"No subject noun after determiner: {prefix!r}")
    span = (start, start + word.end())
    found = prefix[span[0] : span[1]]
    if found != subject:
        raise ValueError(f"Subject mismatch: expected {subject!r}, found {found!r}")
    return span


def last_attractor_span(prefix: str, expected_count: int) -> tuple[int, int]:
    matches = list(ATTRACTOR_RE.finditer(prefix))
    if len(matches) != expected_count:
        raise ValueError(
            f"Expected {expected_count} attractors, matched {len(matches)} in {prefix!r}"
        )
    return matches[-1].span(1)


def span_to_token_index(offsets, span: tuple[int, int], shift: int) -> int:
    """Last token whose character span overlaps the word, plus the BOS shift."""
    start, end = span
    hits = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offsets)
        if tok_start < end and tok_end > start
    ]
    if not hits:
        raise ValueError(f"No token overlaps span {span}")
    return hits[-1] + shift


def bos_shift(tokenizer, token_ids: list[int]) -> int:
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        return 0
    if not token_ids or token_ids[0] != bos_id:
        return 1
    return 0


def annotate_file(src: Path, dst: Path, tokenizer, nua: int) -> dict:
    rows = [json.loads(line) for line in src.open()]
    dst.parent.mkdir(parents=True, exist_ok=True)

    stats = {"rows": len(rows), "p4_is_last": 0, "p0_eq_p4": 0}
    with dst.open("w") as out:
        for idx, row in enumerate(rows):
            record = dict(row)
            for side, sent_key, verb_key, subj_key in (
                ("base", "base_sentence", "MV_base", "MS_base"),
                ("src", "source_sentence", "MV_source", "MS_source"),
            ):
                prefix = context_before_final_verb(row[sent_key], row[verb_key])
                encoded = tokenizer(
                    prefix, add_special_tokens=False, return_offsets_mapping=True
                )
                ids = encoded["input_ids"]
                offsets = encoded["offset_mapping"]
                shift = bos_shift(tokenizer, ids)

                try:
                    p0 = span_to_token_index(
                        offsets, subject_span(prefix, row[subj_key]), shift
                    )
                    p4 = span_to_token_index(
                        offsets, last_attractor_span(prefix, nua), shift
                    )
                except ValueError as exc:
                    raise ValueError(f"{src.name} row {idx} ({side}): {exc}") from exc

                last_index = len(ids) - 1 + shift
                if not 0 <= p0 < p4 < last_index:
                    raise ValueError(
                        f"{src.name} row {idx} ({side}): bad ordering "
                        f"p0={p0} p4={p4} last={last_index}"
                    )
                if p4 == last_index - 1:
                    stats["p4_is_last"] += 1
                record[f"pos_p0_{side}"] = p0
                record[f"pos_p4_{side}"] = p4
                record[f"pos_last_{side}"] = last_index
            out.write(json.dumps(record) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="../data/data_generators/templates/data",
        help="Root of the untouched synthetic dataset tree.",
    )
    parser.add_argument(
        "--out-root",
        default="../data/data_generators/templates/data_positions",
        help="Root for annotated copies; one subdirectory per model tag.",
    )
    parser.add_argument("--model", required=True, help="HF model path for the tokenizer.")
    parser.add_argument("--model-tag", required=True, help="Output subdirectory name.")
    parser.add_argument("--nua", type=int, default=4)
    parser.add_argument(
        "--datasets", nargs="+", default=["obj_rel_across_anim", "obj_rel_across_anim_2"]
    )
    parser.add_argument("--variations", nargs="+", default=["linear", "bow"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_root = (script_dir / args.data_root).resolve()
    out_root = (script_dir / args.out_root).resolve() / args.model_tag

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit(f"{args.model} has no fast tokenizer; offsets unavailable.")
    print(f"[tokenizer] {args.model}  bos_token_id={tokenizer.bos_token_id}")

    total = 0
    for dataset in args.datasets:
        for split in args.splits:
            for variation in args.variations:
                name = f"{dataset}_pairs_{variation}_{args.nua}.jsonl"
                src = data_root / dataset / split / variation / name
                if not src.exists():
                    print(f"[skip] missing {src}")
                    continue
                dst = out_root / dataset / split / variation / name
                stats = annotate_file(src, dst, tokenizer, args.nua)
                total += stats["rows"]
                print(
                    f"[ok] {dataset}/{split}/{variation}  rows={stats['rows']:6d}  "
                    f"p4 immediately before final token: {stats['p4_is_last']}"
                )
    print(f"[done] {total} rows annotated -> {out_root}")


if __name__ == "__main__":
    main()
