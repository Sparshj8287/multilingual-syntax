#!/usr/bin/env python3
import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw model accuracy on obj_rel_across_anim JSONL files "
            "with renormalized two-candidate probabilities."
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


def format_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


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


def render_candidate_block(
    sample_number: int,
    side: str,
    prompt: str,
    cand_a_word: str,
    cand_b_word: str,
    cand_a_info: dict[str, Any],
    cand_b_info: dict[str, Any],
    scores: dict[str, float],
) -> list[str]:
    preferred = cand_a_word if scores["renorm_a"] > scores["renorm_b"] else cand_b_word
    lines = [
        f"=== Sample {sample_number} ({side}) ===",
        "Prompt:",
        prompt,
        "",
        "Candidate tokenization:",
        (
            f"- {cand_a_word}: selected surface='{cand_a_info['selected_surface']}', "
            f"token_ids={cand_a_info['token_ids']}, "
            f"exact_single_token={cand_a_info['exact_single_token']}"
        ),
        (
            f"- {cand_b_word}: selected surface='{cand_b_info['selected_surface']}', "
            f"token_ids={cand_b_info['token_ids']}, "
            f"exact_single_token={cand_b_info['exact_single_token']}"
        ),
        "",
        "Raw next-token probabilities (full vocabulary):",
        f"- P({cand_a_word}) = {scores['raw_a']:.50f}",
        f"- P({cand_b_word}) = {scores['raw_b']:.50f}",
        "",
        "Renormalized over the two candidates only:",
        (
            f"- P({cand_a_word} | {cand_a_word} vs {cand_b_word}) = "
            f"{scores['renorm_a']:.6f}"
        ),
        (
            f"- P({cand_b_word} | {cand_a_word} vs {cand_b_word}) = "
            f"{scores['renorm_b']:.6f}"
        ),
        "",
        f"Preferred candidate: {preferred}",
        "",
        "--------------------------------",
        "",
    ]
    return lines


def build_sample_lines(samples: list[dict[str, Any]]) -> list[str]:
    lines = ["--- Random 50 Sample Logs ---", ""]
    sample_counter = 1
    for sample in samples:
        lines.extend(
            render_candidate_block(
                sample_number=sample_counter,
                side="base",
                prompt=sample["base_context"],
                cand_a_word=sample["mv_base"],
                cand_b_word=sample["mv_source"],
                cand_a_info=sample["mv_base_info"],
                cand_b_info=sample["mv_source_info"],
                scores=sample["base_scores"],
            )
        )
        sample_counter += 1
        lines.extend(
            render_candidate_block(
                sample_number=sample_counter,
                side="source",
                prompt=sample["source_context"],
                cand_a_word=sample["mv_source"],
                cand_b_word=sample["mv_base"],
                cand_a_info=sample["mv_source_info"],
                cand_b_info=sample["mv_base_info"],
                scores=sample["source_scores"],
            )
        )
        sample_counter += 1
    return lines


def write_results_file(
    output_path: Path,
    total_entries: int,
    base_correct: int,
    source_correct: int,
    base_ties: int,
    source_ties: int,
    sample_lines: list[str],
) -> None:
    base_acc = format_accuracy(base_correct, total_entries)
    source_acc = format_accuracy(source_correct, total_entries)

    lines = [
        f"total entries: {total_entries}",
        f"base_overall_accuracy: {base_acc:.6f}",
        f"source_overall_accuracy: {source_acc:.6f}",
        f"base_correct: {base_correct}",
        f"source_correct: {source_correct}",
        f"base_ties: {base_ties}",
        f"source_ties: {source_ties}",
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

    input_root = resolve_path(config_dir, eval_cfg["input_root"])
    output_root = resolve_path(config_dir, eval_cfg.get("output_root", "results"))
    dataset_name = eval_cfg.get("dataset_name", input_root.name)
    variations = eval_cfg.get("input_variations", ["linear", "bow", "maximal"])
    input_glob = eval_cfg.get("input_glob", "*.jsonl")
    sample_logs_per_file = int(eval_cfg.get("sample_logs_per_file", 50))
    seed = int(eval_cfg.get("random_seed", 13))
    show_progress = bool(eval_cfg.get("show_progress_bar", True))

    rng = random.Random(seed)

    jsonl_paths: list[Path] = []
    for variation in variations:
        variation_dir = input_root / variation
        if not variation_dir.exists():
            print(f"Skipping missing variation directory: {variation_dir}")
            continue
        jsonl_paths.extend(sorted(variation_dir.glob(input_glob)))

    if not jsonl_paths:
        raise ValueError(f"No JSONL files found under {input_root}")

    jsonl_counts = {path: count_jsonl_entries(path) for path in jsonl_paths}
    total_examples = sum(jsonl_counts.values()) * len(model_cfgs)

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

        for jsonl_path in jsonl_paths:
            rows = load_jsonl(jsonl_path)
            total_entries = len(rows)
            if total_entries == 0:
                continue

            variation = jsonl_path.parent.name
            desc = f"{model_name}:{variation}/{jsonl_path.stem}"
            iterator = maybe_tqdm(
                enumerate(rows),
                total=total_entries,
                desc=desc,
                enabled=show_progress,
            )

            base_correct = 0
            source_correct = 0
            base_ties = 0
            source_ties = 0
            sampler = ReservoirSampler(sample_logs_per_file, rng)

            for row_idx, row in iterator:
                mv_base = str(row["MV_base"]).strip()
                mv_source = str(row["MV_source"]).strip()
                base_sentence = str(row["base_sentence"]).strip()
                source_sentence = str(row["source_sentence"]).strip()

                mv_base_info = strict_single_token(
                    tokenizer,
                    mv_base,
                    model_name=model_name,
                    jsonl_path=jsonl_path,
                    row_idx=row_idx,
                    field_name="MV_base",
                )
                mv_source_info = strict_single_token(
                    tokenizer,
                    mv_source,
                    model_name=model_name,
                    jsonl_path=jsonl_path,
                    row_idx=row_idx,
                    field_name="MV_source",
                )

                base_context = context_before_final_verb(base_sentence, mv_base)

                source_context = context_before_final_verb(source_sentence, mv_source)


                base_scores = score_two_candidates(
                    model,
                    tokenizer,
                    model_device,
                    base_context,
                    mv_base_info["token_id"],
                    mv_source_info["token_id"],
                )
                source_scores = score_two_candidates(
                    model,
                    tokenizer,
                    model_device,
                    source_context,
                    mv_source_info["token_id"],
                    mv_base_info["token_id"],
                )

                if base_scores["renorm_a"] > base_scores["renorm_b"]:
                    base_correct += 1
                elif base_scores["renorm_a"] == base_scores["renorm_b"]:
                    base_ties += 1

                if source_scores["renorm_a"] > source_scores["renorm_b"]:
                    source_correct += 1
                elif source_scores["renorm_a"] == source_scores["renorm_b"]:
                    source_ties += 1

                sampler.add(
                    {
                        "mv_base": mv_base,
                        "mv_source": mv_source,
                        "mv_base_info": mv_base_info,
                        "mv_source_info": mv_source_info,
                        "base_context": base_context,
                        "source_context": source_context,
                        "base_scores": base_scores,
                        "source_scores": source_scores,
                    }
                )

                if overall_bar is not None:
                    overall_bar.update(1)

            rel_variation = jsonl_path.parent.name
            output_dir = output_root / model_name / dataset_name / rel_variation
            output_path = output_dir / f"{jsonl_path.stem}.txt"
            sample_lines = build_sample_lines(sampler.samples)
            write_results_file(
                output_path=output_path,
                total_entries=total_entries,
                base_correct=base_correct,
                source_correct=source_correct,
                base_ties=base_ties,
                source_ties=source_ties,
                sample_lines=sample_lines,
            )
            print(f"Wrote results: {output_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if overall_bar is not None:
        overall_bar.close()


if __name__ == "__main__":
    main()
