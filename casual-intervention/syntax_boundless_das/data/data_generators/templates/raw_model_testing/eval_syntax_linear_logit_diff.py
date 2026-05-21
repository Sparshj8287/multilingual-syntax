#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-file average logit differences between syntax and "
            "linear/bow hypothesis verbs on paired obj_rel_across_anim test data."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
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


def count_jsonl_entries(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


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


def strict_single_token(
    tokenizer: AutoTokenizer,
    candidate: str,
    *,
    model_name: str,
    jsonl_path: Path,
    row_idx: int,
    field_name: str,
) -> dict[str, Any]:
    word = candidate.strip()
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            "Expected single-token main verb candidate. "
            f"model={model_name}, file={jsonl_path}, row={row_idx}, field={field_name}, "
            f"word='{word}', token_ids={token_ids}"
        )
    return {
        "selected_surface": word,
        "token_ids": token_ids,
        "token_id": token_ids[0],
        "exact_single_token": True,
    }


def context_before_final_verb(sentence: str, final_verb: str) -> str:
    text = sentence.strip()
    if text.endswith("."):
        text = text[:-1]

    suffix = f" {final_verb.strip()}"
    if text.endswith(suffix):
        return text[: -len(suffix)] + " "

    prefix, sep, _last = text.rpartition(" ")
    if not sep:
        raise ValueError(f"Could not build context from sentence: {sentence}")
    return prefix + " "


def score_single_candidate_logit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_device: torch.device,
    context: str,
    candidate_token_id: int,
) -> float:
    model_inputs = tokenizer(context, return_tensors="pt")
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs)
    logits = outputs.logits[0, -1, :]
    return logits[candidate_token_id].item()


class ReservoirSampler:
    def __init__(self, k: int, rng: random.Random):
        self.k = max(0, int(k))
        self.rng = rng
        self.count = 0
        self.samples: list[dict[str, Any]] = []

    def add(self, item: dict[str, Any]) -> None:
        if self.k <= 0:
            return
        self.count += 1
        if len(self.samples) < self.k:
            self.samples.append(item)
            return
        idx = self.rng.randint(1, self.count)
        if idx <= self.k:
            self.samples[idx - 1] = item


def maybe_tqdm(iterable, *, total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def extract_attractor_id(path: Path) -> str:
    match = re.search(r"_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Could not extract attractor index from file name: {path}")
    return match.group(1)


def default_hypothesis_root(syntax_root: Path) -> Path:
    # Typical layout:
    # .../data/obj_rel_across_anim/test -> .../data/obj_rel_across_anim_2/test
    if syntax_root.name == "test":
        dataset_dir = syntax_root.parent
        return dataset_dir.parent / f"{dataset_dir.name}_2" / "test"
    return syntax_root.parent / f"{syntax_root.name}_2"


def collect_paths_by_index(root: Path, variation: str, input_glob: str) -> dict[str, Path]:
    variation_dir = root / variation
    if not variation_dir.exists():
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(variation_dir.glob(input_glob)):
        idx = extract_attractor_id(path)
        indexed[idx] = path
    return indexed


def build_pair_entries(
    syntax_root: Path, hypothesis_root: Path, variations: list[str], input_glob: str
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for variation in variations:
        syntax_map = collect_paths_by_index(syntax_root, variation, input_glob)
        hypothesis_map = collect_paths_by_index(hypothesis_root, variation, input_glob)
        if not syntax_map:
            print(f"Skipping variation with no syntax files: {syntax_root / variation}")
            continue
        if not hypothesis_map:
            print(
                "Skipping variation with no hypothesis files: "
                f"{hypothesis_root / variation}"
            )
            continue

        common = sorted(set(syntax_map) & set(hypothesis_map), key=lambda x: int(x))
        if not common:
            print(
                "Skipping variation with no common attractor ids between "
                f"{syntax_root / variation} and {hypothesis_root / variation}"
            )
            continue

        for idx in common:
            pairs.append(
                {
                    "variation": variation,
                    "attractor_id": idx,
                    "syntax_path": syntax_map[idx],
                    "hypothesis_path": hypothesis_map[idx],
                }
            )
    return pairs


def build_sample_lines(samples: list[dict[str, Any]]) -> list[str]:
    lines = ["--- Random 50 Sample Logs ---", ""]
    for sample_idx, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"=== Sample {sample_idx} ===",
                f"Pair row index: {sample['row_idx']}",
                "",
                f"base_context: {sample['base_context']}",
                f"base_syntax_word: {sample['base_syntax_word']}",
                f"base_hypothesis_word: {sample['base_hypothesis_word']}",
                f"base_syntax_logit: {sample['base_syntax_logit']:.10f}",
                f"base_hypothesis_logit: {sample['base_hypothesis_logit']:.10f}",
                f"base_logit_diff (syntax-hypothesis): {sample['base_diff']:.10f}",
                "",
                f"source_context: {sample['source_context']}",
                f"source_syntax_word: {sample['source_syntax_word']}",
                f"source_hypothesis_word: {sample['source_hypothesis_word']}",
                f"source_syntax_logit: {sample['source_syntax_logit']:.10f}",
                f"source_hypothesis_logit: {sample['source_hypothesis_logit']:.10f}",
                f"source_logit_diff (syntax-hypothesis): {sample['source_diff']:.10f}",
                "",
                "--------------------------------",
                "",
            ]
        )
    return lines


def write_results_file(
    output_path: Path,
    total_entries: int,
    avg_base_diff: float,
    avg_source_diff: float,
    sample_lines: list[str],
) -> None:
    lines = [
        f"total entries: {total_entries}",
        f"base_avg_logit_diff (syntax-hypothesis): {avg_base_diff:.10f}",
        f"source_avg_logit_diff (syntax-hypothesis): {avg_source_diff:.10f}",
        "",
    ]
    lines.extend(sample_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    eval_cfg = config.get("evaluation", {})
    inf_cfg = config.get("inference", {})
    model_cfgs = config.get("models", [])

    if not model_cfgs:
        raise ValueError("Config must include a non-empty models list.")

    syntax_input_root = resolve_path(config_dir, eval_cfg["input_root"])
    hypothesis_input_root = resolve_path(
        config_dir,
        eval_cfg.get("comparison_input_root", str(default_hypothesis_root(syntax_input_root))),
    )
    output_root = resolve_path(config_dir, eval_cfg.get("output_root", "results"))
    dataset_name = eval_cfg.get("dataset_name", syntax_input_root.name)
    input_variations = eval_cfg.get("input_variations", ["linear", "bow"])
    input_glob = eval_cfg.get("input_glob", "*.jsonl")
    sample_logs_per_file = int(eval_cfg.get("sample_logs_per_file", 50))
    seed = int(eval_cfg.get("random_seed", 13))
    show_progress = bool(eval_cfg.get("show_progress_bar", True))

    rng = random.Random(seed)

    pair_entries = build_pair_entries(
        syntax_root=syntax_input_root,
        hypothesis_root=hypothesis_input_root,
        variations=input_variations,
        input_glob=input_glob,
    )
    if not pair_entries:
        raise ValueError(
            "No paired JSONL files found between "
            f"{syntax_input_root} and {hypothesis_input_root}"
        )

    pair_counts = {entry["syntax_path"]: count_jsonl_entries(entry["syntax_path"]) for entry in pair_entries}
    total_examples = sum(pair_counts.values()) * len(model_cfgs)

    device_map = resolve_device_map(
        inf_cfg.get("device_map", "auto"), inf_cfg.get("cuda_device")
    )
    dtype = resolve_dtype(inf_cfg.get("torch_dtype", "bfloat16"))
    dtype = adjust_dtype_for_hardware(dtype)
    use_fast = bool(inf_cfg.get("use_fast_tokenizer", True))

    overall_bar = None
    if show_progress and tqdm is not None:
        overall_bar = tqdm(total=total_examples, desc="Overall progress", dynamic_ncols=True)

    for model_entry in model_cfgs:
        model_name = model_entry["name"]
        checkpoint = model_entry["checkpoint"]
        print(f"\n=== Loading model: {model_name} ({checkpoint}) ===")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=use_fast)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map=device_map,
            torch_dtype=dtype,
        )
        model.eval()
        model_device = next(model.parameters()).device

        for entry in pair_entries:
            variation = entry["variation"]
            attractor_id = entry["attractor_id"]
            syntax_path = entry["syntax_path"]
            hypothesis_path = entry["hypothesis_path"]

            syntax_rows = load_jsonl(syntax_path)
            hypothesis_rows = load_jsonl(hypothesis_path)

            if len(syntax_rows) != len(hypothesis_rows):
                raise ValueError(
                    "Paired files have different lengths: "
                    f"{syntax_path} ({len(syntax_rows)}) vs "
                    f"{hypothesis_path} ({len(hypothesis_rows)})"
                )

            total_entries = len(syntax_rows)
            if total_entries == 0:
                continue

            desc = f"{model_name}:{variation}/att_{attractor_id}"
            iterator = maybe_tqdm(
                enumerate(zip(syntax_rows, hypothesis_rows)),
                total=total_entries,
                desc=desc,
                enabled=show_progress,
            )

            total_base_diff = 0.0
            total_source_diff = 0.0
            sampler = ReservoirSampler(sample_logs_per_file, rng)

            for row_idx, (syntax_row, hypothesis_row) in iterator:
                syntax_mv_base = str(syntax_row["MV_base"]).strip()
                syntax_mv_source = str(syntax_row["MV_source"]).strip()
                hypothesis_mv_base = str(hypothesis_row["MV_base"]).strip()
                hypothesis_mv_source = str(hypothesis_row["MV_source"]).strip()

                syntax_base_sentence = str(syntax_row["base_sentence"]).strip()
                syntax_source_sentence = str(syntax_row["source_sentence"]).strip()

                syntax_mv_base_info = strict_single_token(
                    tokenizer,
                    syntax_mv_base,
                    model_name=model_name,
                    jsonl_path=syntax_path,
                    row_idx=row_idx,
                    field_name="MV_base",
                )
                syntax_mv_source_info = strict_single_token(
                    tokenizer,
                    syntax_mv_source,
                    model_name=model_name,
                    jsonl_path=syntax_path,
                    row_idx=row_idx,
                    field_name="MV_source",
                )
                hypothesis_mv_base_info = strict_single_token(
                    tokenizer,
                    hypothesis_mv_base,
                    model_name=model_name,
                    jsonl_path=hypothesis_path,
                    row_idx=row_idx,
                    field_name="MV_base",
                )
                hypothesis_mv_source_info = strict_single_token(
                    tokenizer,
                    hypothesis_mv_source,
                    model_name=model_name,
                    jsonl_path=hypothesis_path,
                    row_idx=row_idx,
                    field_name="MV_source",
                )

                base_context = context_before_final_verb(syntax_base_sentence, syntax_mv_base)
                source_context = context_before_final_verb(
                    syntax_source_sentence, syntax_mv_source
                )

                base_syntax_logit = score_single_candidate_logit(
                    model,
                    tokenizer,
                    model_device,
                    base_context,
                    syntax_mv_base_info["token_id"],
                )
                base_hypothesis_logit = score_single_candidate_logit(
                    model,
                    tokenizer,
                    model_device,
                    base_context,
                    hypothesis_mv_base_info["token_id"],
                )
                source_syntax_logit = score_single_candidate_logit(
                    model,
                    tokenizer,
                    model_device,
                    source_context,
                    syntax_mv_source_info["token_id"],
                )
                source_hypothesis_logit = score_single_candidate_logit(
                    model,
                    tokenizer,
                    model_device,
                    source_context,
                    hypothesis_mv_source_info["token_id"],
                )

                base_diff = base_syntax_logit - base_hypothesis_logit
                source_diff = source_syntax_logit - source_hypothesis_logit
                total_base_diff += base_diff
                total_source_diff += source_diff

                sampler.add(
                    {
                        "row_idx": row_idx,
                        "base_context": base_context,
                        "source_context": source_context,
                        "base_syntax_word": syntax_mv_base,
                        "base_hypothesis_word": hypothesis_mv_base,
                        "source_syntax_word": syntax_mv_source,
                        "source_hypothesis_word": hypothesis_mv_source,
                        "base_syntax_logit": base_syntax_logit,
                        "base_hypothesis_logit": base_hypothesis_logit,
                        "source_syntax_logit": source_syntax_logit,
                        "source_hypothesis_logit": source_hypothesis_logit,
                        "base_diff": base_diff,
                        "source_diff": source_diff,
                    }
                )

                if overall_bar is not None:
                    overall_bar.update(1)

            avg_base_diff = total_base_diff / total_entries
            avg_source_diff = total_source_diff / total_entries

            output_dir = output_root / model_name / f"{dataset_name}_syntax_vs_hypothesis" / variation
            output_name = f"attractor_{attractor_id}.txt"
            output_path = output_dir / output_name
            sample_lines = build_sample_lines(sampler.samples)
            write_results_file(
                output_path=output_path,
                total_entries=total_entries,
                avg_base_diff=avg_base_diff,
                avg_source_diff=avg_source_diff,
                sample_lines=sample_lines,
            )
            print(
                "Wrote results: "
                f"{output_path} | base_avg_diff={avg_base_diff:.6f}, "
                f"source_avg_diff={avg_source_diff:.6f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if overall_bar is not None:
        overall_bar.close()


if __name__ == "__main__":
    main()
