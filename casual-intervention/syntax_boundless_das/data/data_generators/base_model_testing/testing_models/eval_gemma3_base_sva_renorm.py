#!/usr/bin/env python3
import argparse
import json
import random
import time
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
            "Evaluate subject-verb agreement on base Gemma-3 checkpoints "
            "using renormalized two-candidate next-token probabilities."
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
            "inference.torch_dtype must be one of: float32/fp32, float16/fp16, bfloat16/bf16."
        )
    return mapping[normalized]


def adjust_dtype_for_hardware(dtype: torch.dtype) -> torch.dtype:
    if torch.cuda.is_available():
        return dtype
    if dtype in {torch.float16, torch.bfloat16}:
        print("CPU execution detected; overriding torch_dtype to float32.")
        return torch.float32
    return dtype


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


def pick_single_token_candidate(
    tokenizer: AutoTokenizer, candidate: str
) -> dict[str, Any]:
    base = candidate.strip()
    tried: list[tuple[str, list[int]]] = []
    for surface in (f" {base}", base):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        tried.append((surface, ids))
        if len(ids) == 1:
            return {
                "selected_surface": surface,
                "token_ids": ids,
                "token_id": ids[0],
                "exact_single_token": True,
            }
    non_empty = [(surface, ids) for surface, ids in tried if ids]
    if non_empty:
        best_surface, best_ids = min(non_empty, key=lambda item: len(item[1]))
        return {
            "selected_surface": best_surface,
            "token_ids": best_ids,
            "token_id": best_ids[0],
            "exact_single_token": False,
        }
    return {
        "selected_surface": base,
        "token_ids": [],
        "token_id": None,
        "exact_single_token": False,
    }


class SimpleProgressBar:
    def __init__(
        self,
        total: int,
        desc: str,
        enabled: bool,
        update_every: int = 50,
    ):
        self.total = max(0, int(total))
        self.desc = desc
        self.enabled = enabled
        self.update_every = max(1, int(update_every))
        self.current = 0
        self.start = time.time()
        self.postfix = ""
        if self.enabled:
            print(f"{self.desc}: 0/{self.total}")

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self.current += n
        should_print = (
            self.current >= self.total
            or self.current % self.update_every == 0
            or self.current == 1
        )
        if not should_print:
            return
        elapsed = time.time() - self.start
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else float("inf")
        eta_text = f"{eta:.1f}s" if eta != float("inf") else "unknown"
        pct = (self.current / self.total * 100.0) if self.total else 100.0
        postfix = f" | {self.postfix}" if self.postfix else ""
        print(
            f"{self.desc}: {self.current}/{self.total} "
            f"({pct:.1f}%) eta={eta_text}{postfix}"
        )

    def set_postfix_str(self, postfix: str) -> None:
        self.postfix = postfix

    def close(self) -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self.start
        print(f"{self.desc}: done in {elapsed:.1f}s")


def make_progress_bar(
    total: int,
    desc: str,
    enabled: bool,
    mininterval_sec: float,
    leave: bool = True,
    position: int = 0,
):
    if not enabled:
        return SimpleProgressBar(total=total, desc=desc, enabled=False)
    if tqdm is None:
        return SimpleProgressBar(total=total, desc=desc, enabled=True)
    return tqdm(
        total=total,
        desc=desc,
        leave=leave,
        position=position,
        mininterval=mininterval_sec,
        dynamic_ncols=True,
        unit="ex",
    )


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


def replace_final_verb(sentence: str, old_verb: str, new_verb: str) -> str:
    text = sentence.strip()
    has_period = text.endswith(".")
    core = text[:-1] if has_period else text

    suffix = f" {old_verb.strip()}"
    if core.endswith(suffix):
        updated = core[: -len(suffix)] + f" {new_verb.strip()}"
    else:
        prefix, sep, _last = core.rpartition(" ")
        if not sep:
            raise ValueError(f"Could not replace final verb in sentence: {sentence}")
        updated = prefix + f" {new_verb.strip()}"

    return updated + ("." if has_period else "")


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


def init_eval_metrics() -> dict[str, int]:
    return {
        "base_correct": 0,
        "source_correct": 0,
        "base_ties": 0,
        "source_ties": 0,
    }


def update_side_metrics(
    metrics: dict[str, int], side: str, prob_correct: float, prob_wrong: float
) -> str:
    if prob_correct > prob_wrong:
        metrics[f"{side}_correct"] += 1
        return "correct"
    if prob_correct == prob_wrong:
        metrics[f"{side}_ties"] += 1
        return "tie"
    return "wrong"


class ReservoirSampler:
    def __init__(self, k: int, rng: random.Random):
        self.k = k
        self.rng = rng
        self.count = 0
        self.samples: list[dict[str, Any]] = []

    def add(self, item: dict[str, Any]) -> None:
        self.count += 1
        if len(self.samples) < self.k:
            self.samples.append(item)
            return
        idx = self.rng.randint(1, self.count)
        if idx <= self.k:
            self.samples[idx - 1] = item


def format_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def greedy_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
    input_len = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids = output_ids[0, input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def render_candidate_block(
    sample_number: int,
    side: str,
    prompt: str,
    cand_a_word: str,
    cand_b_word: str,
    cand_a_info: dict[str, Any],
    cand_b_info: dict[str, Any],
    scores: dict[str, float],
    completion: str | None,
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
    ]
    if completion is not None:
        lines.extend(["", "Greedy completion:", completion])
    lines.extend(["", "--------------------------------", ""])
    return lines


def build_per_file_sample_lines(
    samples: list[dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_device: torch.device,
    include_greedy_completion: bool,
    max_new_tokens: int,
) -> list[str]:
    lines: list[str] = []
    sample_counter = 1
    for sample in samples:
        base_completion = (
            greedy_completion(
                model,
                tokenizer,
                model_device,
                sample["base_context"],
                max_new_tokens=max_new_tokens,
            )
            if include_greedy_completion
            else None
        )
        lines.extend(
            render_candidate_block(
                sample_number=sample_counter,
                side="base",
                prompt=sample["base_context"],
                cand_a_word=sample["mv_base"],
                cand_b_word=sample["mv_source"],
                cand_a_info=sample["mv_base_info"],
                cand_b_info=sample["mv_source_info"],
                scores=sample["base_main_scores"],
                completion=base_completion,
            )
        )
        sample_counter += 1

        source_completion = (
            greedy_completion(
                model,
                tokenizer,
                model_device,
                sample["source_context"],
                max_new_tokens=max_new_tokens,
            )
            if include_greedy_completion
            else None
        )
        lines.extend(
            render_candidate_block(
                sample_number=sample_counter,
                side="source",
                prompt=sample["source_context"],
                cand_a_word=sample["mv_source"],
                cand_b_word=sample["mv_base"],
                cand_a_info=sample["mv_source_info"],
                cand_b_info=sample["mv_base_info"],
                scores=sample["source_main_scores"],
                completion=source_completion,
            )
        )
        sample_counter += 1
    return lines


def build_filter2_sample_lines(
    label: str,
    samples: list[dict[str, Any]],
) -> list[str]:
    lines = [f"--- Filter 2 Baseline Samples ({label}) ---", ""]
    if not samples:
        lines.extend(["No samples collected.", ""])
        return lines

    for idx, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"=== Filter2 Sample {idx} ===",
                f"line_index: {sample['line_index']}",
                f"pass_filter2: {sample['pass_filter2']}",
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
    return lines


def write_results_file(
    output_path: Path,
    jsonl_path: Path,
    total_entries: int,
    discarded_filter1: int,
    discarded_filter2_main: int,
    valid_tested_entries_main: int,
    discarded_filter2_aux: int,
    valid_tested_entries_aux: int,
    main_metrics: dict[str, int],
    aux_metrics: dict[str, int],
    filter2_main_sample_lines: list[str],
    filter2_aux_sample_lines: list[str],
    sample_lines: list[str],
) -> None:
    base_main_acc = format_accuracy(
        main_metrics["base_correct"], valid_tested_entries_main
    )
    source_main_acc = format_accuracy(
        main_metrics["source_correct"], valid_tested_entries_main
    )
    base_aux_acc = format_accuracy(aux_metrics["base_correct"], valid_tested_entries_aux)
    source_aux_acc = format_accuracy(
        aux_metrics["source_correct"], valid_tested_entries_aux
    )

    lines = [
        f"jsonl_path: {jsonl_path}",
        f"total_entries: {total_entries}",
        f"discarded_filter1_multitoken: {discarded_filter1}",
        f"discarded_filter2_baseline_fail_main: {discarded_filter2_main}",
        f"valid_tested_entries_main: {valid_tested_entries_main}",
        f"discarded_filter2_baseline_fail_aux: {discarded_filter2_aux}",
        f"valid_tested_entries_aux: {valid_tested_entries_aux}",
        "",
        "--- Main Verb Results ---",
        f"base_overall_accuracy: {base_main_acc:.6f}",
        f"source_overall_accuracy: {source_main_acc:.6f}",
        f"base_correct: {main_metrics['base_correct']}",
        f"source_correct: {main_metrics['source_correct']}",
        f"base_ties: {main_metrics['base_ties']}",
        f"source_ties: {main_metrics['source_ties']}",
        "",
        "--- Auxiliary Verbs Results ---",
        f"base_overall_accuracy: {base_aux_acc:.6f}",
        f"source_overall_accuracy: {source_aux_acc:.6f}",
        f"base_correct: {aux_metrics['base_correct']}",
        f"source_correct: {aux_metrics['source_correct']}",
        f"base_ties: {aux_metrics['base_ties']}",
        f"source_ties: {aux_metrics['source_ties']}",
        "",
    ]
    lines.extend(filter2_main_sample_lines)
    lines.extend(filter2_aux_sample_lines)
    if sample_lines:
        lines.extend(["--- Random 50 Sample Logs ---", ""])
        lines.extend(sample_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_verbose_log(path: Path, samples: list[dict[str, Any]], total_seen: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"total_valid_examples_seen: {total_seen}",
        f"verbose_examples_saved: {len(samples)}",
        "",
    ]

    for idx, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"=== Verbose Sample {idx} ===",
                f"model: {sample['model']}",
                f"jsonl_path: {sample['jsonl_path']}",
                f"line_index: {sample['line_index']}",
                f"variation: {sample['variation']}",
                f"base_sentence: {sample['base_sentence']}",
                f"source_sentence: {sample['source_sentence']}",
                "",
                "[Filter 2 Baseline]",
                f"base_context: {sample['baseline']['base_context']}",
                f"base_correct_verb: {sample['baseline']['base_correct_verb']}",
                f"base_wrong_verb: {sample['baseline']['base_wrong_verb']}",
                f"base_raw_correct: {sample['baseline']['base_raw_correct']:.8f}",
                f"base_raw_wrong: {sample['baseline']['base_raw_wrong']:.8f}",
                f"base_renorm_correct: {sample['baseline']['base_renorm_correct']:.8f}",
                f"base_renorm_wrong: {sample['baseline']['base_renorm_wrong']:.8f}",
                f"source_context: {sample['baseline']['source_context']}",
                f"source_correct_verb: {sample['baseline']['source_correct_verb']}",
                f"source_wrong_verb: {sample['baseline']['source_wrong_verb']}",
                f"source_raw_correct: {sample['baseline']['source_raw_correct']:.8f}",
                f"source_raw_wrong: {sample['baseline']['source_raw_wrong']:.8f}",
                f"source_renorm_correct: {sample['baseline']['source_renorm_correct']:.8f}",
                f"source_renorm_wrong: {sample['baseline']['source_renorm_wrong']:.8f}",
                "",
                "[Eval A - Main Verbs]",
                f"base_context: {sample['main_eval']['base_context']}",
                f"base_correct_verb: {sample['main_eval']['base_correct_verb']}",
                f"base_wrong_verb: {sample['main_eval']['base_wrong_verb']}",
                f"base_raw_correct: {sample['main_eval']['base_raw_correct']:.8f}",
                f"base_raw_wrong: {sample['main_eval']['base_raw_wrong']:.8f}",
                f"base_renorm_correct: {sample['main_eval']['base_renorm_correct']:.8f}",
                f"base_renorm_wrong: {sample['main_eval']['base_renorm_wrong']:.8f}",
                f"source_context: {sample['main_eval']['source_context']}",
                f"source_correct_verb: {sample['main_eval']['source_correct_verb']}",
                f"source_wrong_verb: {sample['main_eval']['source_wrong_verb']}",
                f"source_raw_correct: {sample['main_eval']['source_raw_correct']:.8f}",
                f"source_raw_wrong: {sample['main_eval']['source_raw_wrong']:.8f}",
                f"source_renorm_correct: {sample['main_eval']['source_renorm_correct']:.8f}",
                f"source_renorm_wrong: {sample['main_eval']['source_renorm_wrong']:.8f}",
                "",
                "[Eval B - Auxiliary Pair]",
                f"selected_aux_singular: {sample['aux_eval']['selected_aux_singular']}",
                f"selected_aux_plural: {sample['aux_eval']['selected_aux_plural']}",
                f"base_sentence_replaced: {sample['aux_eval']['base_sentence_replaced']}",
                f"source_sentence_replaced: {sample['aux_eval']['source_sentence_replaced']}",
                f"base_context: {sample['aux_eval']['base_context']}",
                f"base_correct_verb: {sample['aux_eval']['base_correct_verb']}",
                f"base_wrong_verb: {sample['aux_eval']['base_wrong_verb']}",
                f"base_raw_correct: {sample['aux_eval']['base_raw_correct']:.8f}",
                f"base_raw_wrong: {sample['aux_eval']['base_raw_wrong']:.8f}",
                f"base_renorm_correct: {sample['aux_eval']['base_renorm_correct']:.8f}",
                f"base_renorm_wrong: {sample['aux_eval']['base_renorm_wrong']:.8f}",
                f"source_context: {sample['aux_eval']['source_context']}",
                f"source_correct_verb: {sample['aux_eval']['source_correct_verb']}",
                f"source_wrong_verb: {sample['aux_eval']['source_wrong_verb']}",
                f"source_raw_correct: {sample['aux_eval']['source_raw_correct']:.8f}",
                f"source_raw_wrong: {sample['aux_eval']['source_raw_wrong']:.8f}",
                f"source_renorm_correct: {sample['aux_eval']['source_renorm_correct']:.8f}",
                f"source_renorm_wrong: {sample['aux_eval']['source_renorm_wrong']:.8f}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    eval_cfg = config.get("evaluation", {})
    inf_cfg = config.get("inference", {})
    model_cfgs = config.get("models", [])
    aux_pairs = config.get("auxiliary_verb_pairs", [])



    if not model_cfgs:
        raise ValueError("config.yaml must include a non-empty models list.")
    if not aux_pairs:
        raise ValueError("config.yaml must include auxiliary_verb_pairs.")

    input_root = resolve_path(config_dir, eval_cfg["input_root"])
    output_root = resolve_path(config_dir, eval_cfg["output_root"])
    variations = eval_cfg.get("input_variations", ["linear", "bow", "maximal"])
    input_glob = eval_cfg.get("input_glob", "*.jsonl")
    verbose_sample_size = int(eval_cfg.get("verbose_sample_size", 50))
    seed = int(eval_cfg.get("random_seed", 13))
    show_progress_bar = bool(eval_cfg.get("show_progress_bar", True))
    progress_mininterval_sec = float(eval_cfg.get("progress_mininterval_sec", 0.1))
    sample_logs_per_file = int(eval_cfg.get("sample_logs_per_file", 50))
    include_sample_logs_in_results = bool(
        eval_cfg.get("include_sample_logs_in_results", True)
    )
    sample_log_include_greedy_completion = bool(
        eval_cfg.get("sample_log_include_greedy_completion", True)
    )
    sample_log_max_new_tokens = int(eval_cfg.get("sample_log_max_new_tokens", 24))

    rng = random.Random(seed)
    sampler = ReservoirSampler(verbose_sample_size, rng)


    jsonl_paths: list[Path] = []
    for variation in variations:
        variation_dir = input_root / variation
        if not variation_dir.exists():
            print(f"Skipping missing variation directory: {variation_dir}")
            continue
        jsonl_paths.extend(sorted(variation_dir.glob(input_glob)))

    if not jsonl_paths:
        raise ValueError(f"No JSONL files found under {input_root} with variations={variations}")

    jsonl_entry_counts = {path: count_jsonl_entries(path) for path in jsonl_paths}

    examples_per_model = sum(count * 3 for count in jsonl_entry_counts.values())
    total_examples_all_models = examples_per_model * len(model_cfgs)



    device_map = resolve_device_map(
        inf_cfg.get("device_map", "auto"), inf_cfg.get("cuda_device")
    )
    dtype = resolve_dtype(inf_cfg.get("torch_dtype", "bfloat16"))
    dtype = adjust_dtype_for_hardware(dtype)
    use_fast = bool(inf_cfg.get("use_fast_tokenizer", True))
    overall_bar = make_progress_bar(
        total=total_examples_all_models,
        desc="Overall dataset progress",
        enabled=show_progress_bar,
        mininterval_sec=progress_mininterval_sec,
        leave=True,
        position=0,
    )

    try:
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

            model_bar = make_progress_bar(
                total=examples_per_model,
                desc=f"Model {model_name}",
                enabled=show_progress_bar,
                mininterval_sec=progress_mininterval_sec,
                leave=True,
                position=1,
            )

            aux_pair_ids: list[tuple[str, str, int, int]] = []
            skipped_aux_pairs = 0
            for pair in aux_pairs:
                sg, pl = str(pair[0]).strip(), str(pair[1]).strip()
                sg_info = pick_single_token_candidate(tokenizer, sg)
                pl_info = pick_single_token_candidate(tokenizer, pl)
                if (
                    not sg_info["exact_single_token"]
                    or not pl_info["exact_single_token"]
                    or sg_info["token_id"] is None
                    or pl_info["token_id"] is None
                ):
                    skipped_aux_pairs += 1
                    continue
                aux_pair_ids.append((sg, pl, sg_info["token_id"], pl_info["token_id"]))
            if skipped_aux_pairs > 0:
                print(
                    f"Model {model_name}: skipped {skipped_aux_pairs} auxiliary pairs "
                    "that are not single-token."
                )
            if not aux_pair_ids:
                print(
                    f"Model {model_name}: no valid auxiliary pairs available; "
                    "Eval B will be skipped without discarding examples."
                )


            for jsonl_path in jsonl_paths:
                rows = load_jsonl(jsonl_path)
                total_entries = len(rows)
                discarded_filter1 = 0
                discarded_filter2_main = 0
                discarded_filter2_aux = 0
                valid_main = 0
                valid_aux = 0

                main_metrics = init_eval_metrics()
                aux_metrics = init_eval_metrics()
                file_sampler = ReservoirSampler(sample_logs_per_file, rng)
                filter2_main_sampler = ReservoirSampler(5, rng)
                filter2_aux_sampler = ReservoirSampler(5, rng)

                variation = jsonl_path.parent.name
                file_bar = make_progress_bar(
                    total=total_entries * 3,
                    desc=f"{model_name}:{variation}/{jsonl_path.stem}",
                    enabled=show_progress_bar,
                    mininterval_sec=progress_mininterval_sec,
                    leave=False,
                    position=2,
                )
                filtered_rows: list[dict[str, Any]] = []
                for idx, row in enumerate(rows):
                    mv_base = str(row["MV_base"]).strip()
                    mv_source = str(row["MV_source"]).strip()
                    ms_base = str(row["MS_base"]).strip()
                    ms_source = str(row["MS_source"]).strip()
                    base_sentence = str(row["base_sentence"])
                    source_sentence = str(row["source_sentence"])

                    # Filter 1: only check if MV_base and MV_source are single tokens.
                    mv_base_info = pick_single_token_candidate(tokenizer, mv_base)
                    mv_source_info = pick_single_token_candidate(tokenizer, mv_source)
                    if (
                        not mv_base_info["exact_single_token"]
                        or not mv_source_info["exact_single_token"]
                        or mv_base_info["token_id"] is None
                        or mv_source_info["token_id"] is None
                    ):
                        discarded_filter1 += 1
                        file_bar.update(1)
                        model_bar.update(1)
                        overall_bar.update(1)
                        continue

                    filtered_rows.append(
                        {
                            "line_index": idx,
                            "mv_base": mv_base,
                            "mv_source": mv_source,
                            "ms_base": ms_base,
                            "ms_source": ms_source,
                            "base_sentence": base_sentence,
                            "source_sentence": source_sentence,
                            "mv_base_info": mv_base_info,
                            "mv_source_info": mv_source_info,
                            "mv_base_id": mv_base_info["token_id"],
                            "mv_source_id": mv_source_info["token_id"],
                        }
                    )
                    file_bar.update(1)
                    model_bar.update(1)
                    overall_bar.update(1)

                # Account for skipped main-loop steps caused by Filter 1.
                # Aux loop is independent and still runs on all rows.
                skipped_secondary_steps = discarded_filter1
                if skipped_secondary_steps > 0:
                    file_bar.update(skipped_secondary_steps)
                    model_bar.update(skipped_secondary_steps)
                    overall_bar.update(skipped_secondary_steps)

                # Main-verb pipeline (Filter 2 main baseline -> Main evaluation).
                for item in filtered_rows:
                    mv_base = item["mv_base"]
                    mv_source = item["mv_source"]
                    mv_base_id = item["mv_base_id"]
                    mv_source_id = item["mv_source_id"]
                    ms_base = item["ms_base"]
                    ms_source = item["ms_source"]
                    base_sentence = item["base_sentence"]
                    source_sentence = item["source_sentence"]

                    base_baseline_ctx = f"The {ms_base} "
                    source_baseline_ctx = f"The {ms_source} "
                    base_baseline_main = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        base_baseline_ctx,
                        mv_base_id,
                        mv_source_id,
                    )
                    source_baseline_main = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        source_baseline_ctx,
                        mv_source_id,
                        mv_base_id,
                    )
                    main_filter2_pass = (
                        base_baseline_main["renorm_a"] > base_baseline_main["renorm_b"]
                        and source_baseline_main["renorm_a"]
                        > source_baseline_main["renorm_b"]
                    )
                    filter2_main_sampler.add(
                        {
                            "line_index": item["line_index"],
                            "pass_filter2": main_filter2_pass,
                            "base_context": base_baseline_ctx,
                            "base_correct": mv_base,
                            "base_wrong": mv_source,
                            "base_raw_correct": base_baseline_main["raw_a"],
                            "base_raw_wrong": base_baseline_main["raw_b"],
                            "base_renorm_correct": base_baseline_main["renorm_a"],
                            "base_renorm_wrong": base_baseline_main["renorm_b"],
                            "source_context": source_baseline_ctx,
                            "source_correct": mv_source,
                            "source_wrong": mv_base,
                            "source_raw_correct": source_baseline_main["raw_a"],
                            "source_raw_wrong": source_baseline_main["raw_b"],
                            "source_renorm_correct": source_baseline_main["renorm_a"],
                            "source_renorm_wrong": source_baseline_main["renorm_b"],
                        }
                    )
                    if not main_filter2_pass:
                        discarded_filter2_main += 1
                    else:
                        valid_main += 1

                        base_context = context_before_final_verb(base_sentence, mv_base)
                        source_context = context_before_final_verb(
                            source_sentence, mv_source
                        )
                        base_main = score_two_candidates(
                            model,
                            tokenizer,
                            model_device,
                            base_context,
                            mv_base_id,
                            mv_source_id,
                        )
                        source_main = score_two_candidates(
                            model,
                            tokenizer,
                            model_device,
                            source_context,
                            mv_source_id,
                            mv_base_id,
                        )
                        update_side_metrics(
                            main_metrics,
                            "base",
                            base_main["renorm_a"],
                            base_main["renorm_b"],
                        )
                        update_side_metrics(
                            main_metrics,
                            "source",
                            source_main["renorm_a"],
                            source_main["renorm_b"],
                        )
                        file_sampler.add(
                            {
                                "mv_base": mv_base,
                                "mv_source": mv_source,
                                "mv_base_info": item["mv_base_info"],
                                "mv_source_info": item["mv_source_info"],
                                "base_context": base_context,
                                "source_context": source_context,
                                "base_main_scores": base_main,
                                "source_main_scores": source_main,
                            }
                        )
                        sampler.add(
                            {
                                "model": model_name,
                                "jsonl_path": str(jsonl_path),
                                "line_index": item["line_index"],
                                "variation": variation,
                                "base_sentence": base_sentence,
                                "source_sentence": source_sentence,
                                "baseline": {
                                    "base_context": base_baseline_ctx,
                                    "base_correct_verb": mv_base,
                                    "base_wrong_verb": mv_source,
                                    "base_raw_correct": base_baseline_main["raw_a"],
                                    "base_raw_wrong": base_baseline_main["raw_b"],
                                    "base_renorm_correct": base_baseline_main["renorm_a"],
                                    "base_renorm_wrong": base_baseline_main["renorm_b"],
                                    "source_context": source_baseline_ctx,
                                    "source_correct_verb": mv_source,
                                    "source_wrong_verb": mv_base,
                                    "source_raw_correct": source_baseline_main["raw_a"],
                                    "source_raw_wrong": source_baseline_main["raw_b"],
                                    "source_renorm_correct": source_baseline_main["renorm_a"],
                                    "source_renorm_wrong": source_baseline_main["renorm_b"],
                                },
                                "main_eval": {
                                    "base_context": base_context,
                                    "base_correct_verb": mv_base,
                                    "base_wrong_verb": mv_source,
                                    "base_raw_correct": base_main["raw_a"],
                                    "base_raw_wrong": base_main["raw_b"],
                                    "base_renorm_correct": base_main["renorm_a"],
                                    "base_renorm_wrong": base_main["renorm_b"],
                                    "source_context": source_context,
                                    "source_correct_verb": mv_source,
                                    "source_wrong_verb": mv_base,
                                    "source_raw_correct": source_main["raw_a"],
                                    "source_raw_wrong": source_main["raw_b"],
                                    "source_renorm_correct": source_main["renorm_a"],
                                    "source_renorm_wrong": source_main["renorm_b"],
                                },
                                "aux_eval": {
                                    "selected_aux_singular": "",
                                    "selected_aux_plural": "",
                                    "base_sentence_replaced": "",
                                    "source_sentence_replaced": "",
                                    "base_context": "",
                                    "base_correct_verb": "",
                                    "base_wrong_verb": "",
                                    "base_raw_correct": 0.0,
                                    "base_raw_wrong": 0.0,
                                    "base_renorm_correct": 0.0,
                                    "base_renorm_wrong": 0.0,
                                    "source_context": "",
                                    "source_correct_verb": "",
                                    "source_wrong_verb": "",
                                    "source_raw_correct": 0.0,
                                    "source_raw_wrong": 0.0,
                                    "source_renorm_correct": 0.0,
                                    "source_renorm_wrong": 0.0,
                                },
                            }
                        )

                    file_bar.update(1)
                    model_bar.update(1)
                    overall_bar.update(1)

                # Auxiliary-verb pipeline (independent from main Filter 1).
                for idx, row in enumerate(rows):
                    ms_base = str(row["MS_base"]).strip()
                    ms_source = str(row["MS_source"]).strip()
                    mv_base = str(row["MV_base"]).strip()
                    mv_source = str(row["MV_source"]).strip()
                    base_sentence = str(row["base_sentence"])
                    source_sentence = str(row["source_sentence"])

                    if not aux_pair_ids:
                        discarded_filter2_aux += 1
                        filter2_aux_sampler.add(
                            {
                                "line_index": idx,
                                "pass_filter2": False,
                                "base_context": f"The {ms_base} ",
                                "base_correct": "N/A",
                                "base_wrong": "N/A",
                                "base_raw_correct": 0.0,
                                "base_raw_wrong": 0.0,
                                "base_renorm_correct": 0.0,
                                "base_renorm_wrong": 0.0,
                                "source_context": f"The {ms_source} ",
                                "source_correct": "N/A",
                                "source_wrong": "N/A",
                                "source_raw_correct": 0.0,
                                "source_raw_wrong": 0.0,
                                "source_renorm_correct": 0.0,
                                "source_renorm_wrong": 0.0,
                            }
                        )
                        file_bar.update(1)
                        model_bar.update(1)
                        overall_bar.update(1)
                        continue

                    aux_sg, aux_pl, aux_sg_id, aux_pl_id = rng.choice(aux_pair_ids)
                    base_baseline_aux_ctx = f"The {ms_base} "
                    source_baseline_aux_ctx = f"The {ms_source} "
                    base_baseline_aux = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        base_baseline_aux_ctx,
                        aux_sg_id,
                        aux_pl_id,
                    )
                    source_baseline_aux = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        source_baseline_aux_ctx,
                        aux_pl_id,
                        aux_sg_id,
                    )
                    aux_filter2_pass = (
                        base_baseline_aux["renorm_a"] > base_baseline_aux["renorm_b"]
                        and source_baseline_aux["renorm_a"] > source_baseline_aux["renorm_b"]
                    )
                    filter2_aux_sampler.add(
                        {
                            "line_index": idx,
                            "pass_filter2": aux_filter2_pass,
                            "base_context": base_baseline_aux_ctx,
                            "base_correct": aux_sg.strip(),
                            "base_wrong": aux_pl.strip(),
                            "base_raw_correct": base_baseline_aux["raw_a"],
                            "base_raw_wrong": base_baseline_aux["raw_b"],
                            "base_renorm_correct": base_baseline_aux["renorm_a"],
                            "base_renorm_wrong": base_baseline_aux["renorm_b"],
                            "source_context": source_baseline_aux_ctx,
                            "source_correct": aux_pl.strip(),
                            "source_wrong": aux_sg.strip(),
                            "source_raw_correct": source_baseline_aux["raw_a"],
                            "source_raw_wrong": source_baseline_aux["raw_b"],
                            "source_renorm_correct": source_baseline_aux["renorm_a"],
                            "source_renorm_wrong": source_baseline_aux["renorm_b"],
                        }
                    )
                    if not aux_filter2_pass:
                        discarded_filter2_aux += 1
                        file_bar.update(1)
                        model_bar.update(1)
                        overall_bar.update(1)
                        continue

                    valid_aux += 1

                    aux_base_sentence = replace_final_verb(base_sentence, mv_base, aux_sg)
                    aux_source_sentence = replace_final_verb(
                        source_sentence, mv_source, aux_pl
                    )
                    aux_base_context = context_before_final_verb(
                        aux_base_sentence, aux_sg
                    )
                    aux_source_context = context_before_final_verb(
                        aux_source_sentence, aux_pl
                    )
                    base_aux = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        aux_base_context,
                        aux_sg_id,
                        aux_pl_id,
                    )
                    source_aux = score_two_candidates(
                        model,
                        tokenizer,
                        model_device,
                        aux_source_context,
                        aux_pl_id,
                        aux_sg_id,
                    )
                    update_side_metrics(
                        aux_metrics,
                        "base",
                        base_aux["renorm_a"],
                        base_aux["renorm_b"],
                    )
                    update_side_metrics(
                        aux_metrics,
                        "source",
                        source_aux["renorm_a"],
                        source_aux["renorm_b"],
                    )

                    file_bar.update(1)
                    model_bar.update(1)
                    overall_bar.update(1)

                file_bar.close()

                rel_parent = jsonl_path.parent.relative_to(input_root)
                output_dir = output_root / model_name / rel_parent
                output_name = f"{jsonl_path.stem}.txt"
                output_path = output_dir / output_name
                sample_lines: list[str] = []
                if include_sample_logs_in_results and file_sampler.samples:
                    sample_lines = build_per_file_sample_lines(
                        samples=file_sampler.samples,
                        model=model,
                        tokenizer=tokenizer,
                        model_device=model_device,
                        include_greedy_completion=sample_log_include_greedy_completion,
                        max_new_tokens=sample_log_max_new_tokens,
                    )
                filter2_main_sample_lines = build_filter2_sample_lines(
                    label="Main Verbs (5 random)",
                    samples=filter2_main_sampler.samples,
                )
                filter2_aux_sample_lines = build_filter2_sample_lines(
                    label="Auxiliary Verbs (5 random)",
                    samples=filter2_aux_sampler.samples,
                )
                write_results_file(
                    output_path=output_path,
                    jsonl_path=jsonl_path,
                    total_entries=total_entries,
                    discarded_filter1=discarded_filter1,
                    discarded_filter2_main=discarded_filter2_main,
                    valid_tested_entries_main=valid_main,
                    discarded_filter2_aux=discarded_filter2_aux,
                    valid_tested_entries_aux=valid_aux,
                    main_metrics=main_metrics,
                    aux_metrics=aux_metrics,
                    filter2_main_sample_lines=filter2_main_sample_lines,
                    filter2_aux_sample_lines=filter2_aux_sample_lines,
                    sample_lines=sample_lines,
                )
                print(f"Wrote results: {output_path}")

            model_bar.close()

            # Release model memory before loading the next checkpoint.
            del model
            torch.cuda.empty_cache()
    finally:
        overall_bar.close()

    verbose_log_path = resolve_path(
        config_dir,
        eval_cfg.get("verbose_log_path", str(output_root / "verbose_random_50.txt")),
    )
    write_verbose_log(verbose_log_path, sampler.samples, sampler.count)
    print(f"Wrote verbose log: {verbose_log_path}")


if __name__ == "__main__":
    main()
