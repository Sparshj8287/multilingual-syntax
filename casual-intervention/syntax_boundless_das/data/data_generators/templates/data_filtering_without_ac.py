#!/usr/bin/env python3
import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_MODELS = {
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
    "llama-3.1-8b": "/home/models/Llama-3.1-8B",
    "qwen-3-8b": "Qwen/Qwen3-8B-Base",
    "olmo-3-7b": "allenai/Olmo-3-1025-7B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter obj_rel_across_anim JSONL files without attractor context "
            "using sequential multi-model renormalized scoring."
        )
    )
    parser.add_argument(
        "--config",
        default="config_obj_rel_across_anim.yaml",
        help="Path to config file (default: config_obj_rel_across_anim.yaml).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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


def resolve_device_map(
    device_map_cfg: str | None, cuda_device: int | None
) -> str | dict[str, str]:
    if not torch.cuda.is_available():
        if cuda_device is not None:
            print("CUDA is not available; ignoring cuda_device and using CPU.")
        return {"": "cpu"}

    if cuda_device is not None:
        available = torch.cuda.device_count()
        if cuda_device < 0 or cuda_device >= available:
            raise ValueError(
                f"Invalid cuda_device={cuda_device}. Available devices: 0..{available - 1}"
            )
        return {"": f"cuda:{cuda_device}"}

    if not device_map_cfg:
        return "auto"
    if device_map_cfg == "cpu":
        return {"": "cpu"}
    if device_map_cfg.startswith("cuda:"):
        return {"": device_map_cfg}
    return device_map_cfg


def resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = (dtype_name or "bfloat16").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(
            "torch_dtype must be one of: float32/fp32, float16/fp16, bfloat16/bf16."
        )
    return mapping[normalized]


def adjust_dtype_for_hardware(dtype: torch.dtype) -> torch.dtype:
    if torch.cuda.is_available():
        return dtype
    if dtype in {torch.float16, torch.bfloat16}:
        print("CPU execution detected; overriding torch_dtype to float32.")
        return torch.float32
    return dtype


def strict_next_token_id(
    tokenizer: AutoTokenizer,
    candidate: str,
    *,
    model_name: str,
    dataset: str,
    row_idx: int,
    field_name: str,
) -> int:
    base = candidate.strip()
    without_space_ids = tokenizer.encode(base, add_special_tokens=False)
    if len(without_space_ids) == 1:
        return without_space_ids[0]

    raise ValueError(
        "Main verb is not a single next-token candidate. "
        f"model={model_name}, dataset={dataset}, row_idx={row_idx}, field={field_name}, "
        f"verb='{base}', ids_without_space={without_space_ids}"
    )


def score_two_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_device: torch.device,
    context: str,
    candidate_a_id: int,
    candidate_b_id: int,
) -> dict[str, float]:
    model_inputs = tokenizer(context, return_tensors="pt")
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

    with torch.inference_mode():
        outputs = model(**model_inputs)

    logits = outputs.logits[0, -1, :]
    full_probs = torch.softmax(logits, dim=-1)

    raw_a = full_probs[candidate_a_id].item()
    raw_b = full_probs[candidate_b_id].item()

    pair_logits = torch.stack([logits[candidate_a_id], logits[candidate_b_id]])
    pair_probs = torch.softmax(pair_logits, dim=0)

    return {
        "raw_a": raw_a,
        "raw_b": raw_b,
        "renorm_a": pair_probs[0].item(),
        "renorm_b": pair_probs[1].item(),
    }


def maybe_tqdm(
    iterable,
    total: int,
    desc: str,
    enabled: bool,
):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def extract_attractor_count(path: Path) -> int:
    stem = path.stem
    maybe_n = stem.rsplit("_", maxsplit=1)[-1]
    return int(maybe_n)


def sample_or_all(rows: list[dict[str, Any]], k: int, rng: random.Random) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    if len(rows) <= k:
        return list(rows)
    return rng.sample(rows, k)


def build_eval_sample(
    row_idx: int,
    base_context: str,
    source_context: str,
    mv_base: str,
    mv_source: str,
    base_scores: dict[str, float],
    source_scores: dict[str, float],
) -> dict[str, Any]:
    return {
        "row_idx": row_idx,
        "base_context": base_context,
        "base_correct": mv_base,
        "base_wrong": mv_source,
        "base_raw_correct": base_scores["raw_a"],
        "base_raw_wrong": base_scores["raw_b"],
        "base_renorm_correct": base_scores["renorm_a"],
        "base_renorm_wrong": base_scores["renorm_b"],
        "source_context": source_context,
        "source_correct": mv_source,
        "source_wrong": mv_base,
        "source_raw_correct": source_scores["raw_a"],
        "source_raw_wrong": source_scores["raw_b"],
        "source_renorm_correct": source_scores["renorm_a"],
        "source_renorm_wrong": source_scores["renorm_b"],
    }


def format_model_case_section(
    variation: str,
    attractor_n: int,
    model_name: str,
    total_sentences: int,
    base_correct: int,
    source_correct: int,
    filtered_sentences: int,
    samples: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"case: {attractor_n} attractor and model: {model_name}",
        f"dataset: {variation}",
        f"total sentences: {total_sentences}",
        f"base correct : {base_correct}/{total_sentences}",
        f"source correct: {source_correct}/{total_sentences}",
        f"filtered_sentences: {filtered_sentences}",
        "",
    ]
    for sample_id, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"=== Sample {sample_id} ===",
                f"row_index: {sample['row_idx']}",
                f"base_context: {sample['base_context']}",
                f"base_correct: {sample['base_correct']}",
                f"base_wrong: {sample['base_wrong']}",
                f"base_raw_correct: {sample['base_raw_correct']:.8f}",
                f"base_raw_wrong: {sample['base_raw_wrong']:.8f}",
                f"base_renorm_correct: {sample['base_renorm_correct']:.8f}",
                f"base_renorm_wrong: {sample['base_renorm_wrong']:.8f}",
                f"source_context: {sample['source_context']}",
                f"source_correct: {sample['source_correct']}",
                f"source_wrong: {sample['source_wrong']}",
                f"source_raw_correct: {sample['source_raw_correct']:.8f}",
                f"source_raw_wrong: {sample['source_raw_wrong']:.8f}",
                f"source_renorm_correct: {sample['source_renorm_correct']:.8f}",
                f"source_renorm_wrong: {sample['source_renorm_wrong']:.8f}",
                "",
            ]
        )
    lines.append("")
    return lines


def format_final_samples_section(
    file_name: str,
    rows: list[dict[str, Any]],
    max_samples: int,
    rng: random.Random,
) -> list[str]:
    lines = [f"final samples from: {file_name}", f"total_remaining: {len(rows)}", ""]
    chosen = sample_or_all(rows, max_samples, rng)
    for idx, row in enumerate(chosen, start=1):
        lines.extend(
            [
                f"--- Final Sample {idx} ---",
                f"base_sentence: {row.get('base_sentence', '')}",
                f"source_sentence: {row.get('source_sentence', '')}",
                f"MS_base: {row.get('MS_base', '')}",
                f"MV_base: {row.get('MV_base', '')}",
                f"MS_source: {row.get('MS_source', '')}",
                f"MV_source: {row.get('MV_source', '')}",
                "",
            ]
        )
    lines.append("")
    return lines


def get_model_map(filter_cfg: dict[str, Any]) -> dict[str, str]:
    models_cfg = filter_cfg.get("models")
    if not models_cfg:
        return dict(DEFAULT_MODELS)
    if isinstance(models_cfg, dict):
        return {str(k): str(v) for k, v in models_cfg.items()}
    raise ValueError("filtering_without_ac.models must be a mapping of name -> checkpoint.")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    filter_cfg = config.get("filtering_without_ac", {})
    models_to_use = get_model_map(filter_cfg)
    variations = filter_cfg.get("variations", ["linear", "bow", "maximal"])
    sample_per_model = int(filter_cfg.get("samples_per_model_case", 5))
    final_samples_per_file = int(filter_cfg.get("final_samples_per_file", 10))
    seed = int(filter_cfg.get("seed", 13))
    show_progress = bool(filter_cfg.get("show_progress_bar", True))
    use_fast = bool(filter_cfg.get("use_fast_tokenizer", True))

    input_root = resolve_path(
        config_dir,
        filter_cfg.get("input_root", "data/obj_rel_across_anim"),
    )
    output_root = resolve_path(
        config_dir,
        filter_cfg.get("output_root", "data/cleaned_obj_rel_across_anim"),
    )

    device_map = resolve_device_map(
        filter_cfg.get("device_map", "auto"), filter_cfg.get("cuda_device")
    )
    dtype = resolve_dtype(filter_cfg.get("torch_dtype", "bfloat16"))
    dtype = adjust_dtype_for_hardware(dtype)

    rng = random.Random(seed)

    for variation in variations:
        variation_dir = input_root / variation
        if not variation_dir.exists():
            print(f"Skipping missing variation dir: {variation_dir}")
            continue

        jsonl_files = sorted(
            variation_dir.glob("*.jsonl"),
            key=extract_attractor_count,
        )
        if not jsonl_files:
            print(f"No files found in {variation_dir}, skipping.")
            continue

        print(f"\n=== Processing dataset: {variation} ===")
        current_rows_by_file = {path: load_jsonl(path) for path in jsonl_files}
        report_lines = [
            f"dataset: {variation}",
            f"input_root: {variation_dir}",
            f"models: {', '.join(models_to_use.keys())}",
            "",
        ]

        for model_name, checkpoint in models_to_use.items():
            print(f"\nLoading model: {model_name} ({checkpoint})")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                device_map=device_map,
                torch_dtype=dtype,
            )
            model.eval()
            model_device = next(model.parameters()).device

            for jsonl_path in jsonl_files:
                rows = current_rows_by_file[jsonl_path]
                total = len(rows)
                attractor_n = extract_attractor_count(jsonl_path)
                desc = f"{variation} | {model_name} | N={attractor_n}"

                kept_rows: list[dict[str, Any]] = []
                per_case_samples: list[dict[str, Any]] = []
                base_correct = 0
                source_correct = 0

                iterable = maybe_tqdm(
                    enumerate(rows),
                    total=total,
                    desc=desc,
                    enabled=show_progress,
                )
                for row_idx, row in iterable:
                    mv_base = str(row.get("MV_base", "")).strip()
                    mv_source = str(row.get("MV_source", "")).strip()
                    ms_base = str(row.get("MS_base", "")).strip()
                    ms_source = str(row.get("MS_source", "")).strip()

                    if not (mv_base and mv_source and ms_base and ms_source):
                        continue

                    mv_base_id = strict_next_token_id(
                        tokenizer,
                        mv_base,
                        model_name=model_name,
                        dataset=f"{variation}/{jsonl_path.name}",
                        row_idx=row_idx,
                        field_name="MV_base",
                    )
                    mv_source_id = strict_next_token_id(
                        tokenizer,
                        mv_source,
                        model_name=model_name,
                        dataset=f"{variation}/{jsonl_path.name}",
                        row_idx=row_idx,
                        field_name="MV_source",
                    )

                    base_context = f"The {ms_base} "
                    source_context = f"The {ms_source} "

                    base_scores = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        base_context,
                        mv_base_id,
                        mv_source_id,
                    )
                    source_scores = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        source_context,
                        mv_source_id,
                        mv_base_id,
                    )

                    base_ok = base_scores["renorm_a"] > base_scores["renorm_b"]
                    source_ok = source_scores["renorm_a"] > source_scores["renorm_b"]

                    if base_ok:
                        base_correct += 1
                    if source_ok:
                        source_correct += 1

                    per_case_samples.append(
                        build_eval_sample(
                            row_idx=row_idx,
                            base_context=base_context,
                            source_context=source_context,
                            mv_base=mv_base,
                            mv_source=mv_source,
                            base_scores=base_scores,
                            source_scores=source_scores,
                        )
                    )

                    if base_ok and source_ok:
                        kept_rows.append(row)

                current_rows_by_file[jsonl_path] = kept_rows
                sampled = sample_or_all(per_case_samples, sample_per_model, rng)
                report_lines.extend(
                    format_model_case_section(
                        variation=variation,
                        attractor_n=attractor_n,
                        model_name=model_name,
                        total_sentences=total,
                        base_correct=base_correct,
                        source_correct=source_correct,
                        filtered_sentences=len(kept_rows),
                        samples=sampled,
                    )
                )

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_variation_dir = output_root / variation
        out_variation_dir.mkdir(parents=True, exist_ok=True)
        for jsonl_path in jsonl_files:
            out_path = out_variation_dir / jsonl_path.name
            write_jsonl(out_path, current_rows_by_file[jsonl_path])

        report_lines.extend(
            [
                "================ FINAL FILTERED DATA SAMPLES ================",
                "",
            ]
        )
        for jsonl_path in jsonl_files:
            report_lines.extend(
                format_final_samples_section(
                    file_name=jsonl_path.name,
                    rows=current_rows_by_file[jsonl_path],
                    max_samples=final_samples_per_file,
                    rng=rng,
                )
            )

        report_path = output_root / f"{variation}_filter_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Saved cleaned dataset for {variation} at: {out_variation_dir}")
        print(f"Saved report for {variation} at: {report_path}")


if __name__ == "__main__":
    main()
