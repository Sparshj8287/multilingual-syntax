import argparse
import json
import math
import os
import random
import re
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

PROMPT_TEMPLATE = """Complete the following sentence with the correct form of the verb. Please answer in one word:
Sentence: {sentence}
Options: {options}
Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate models on JSONL templates data."
    )
    parser.add_argument(
        "--data-dir",
        default=(
            "casual-intervention/syntax_boundless_das/data/data_generators/templates/data"
        ),
        help="Root directory containing JSONL files.",
    )
    parser.add_argument(
        "--results-dir",
        default=(
            "casual-intervention/syntax_boundless_das/data/data_generators/base_model_testing/results"
        ),
        help="Directory to write results to.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate for samples.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top next-token predictions to include.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random examples to generate per JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def list_jsonl_files(data_dir: str) -> list[str]:
    jsonl_files: list[str] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, name))
    return sorted(jsonl_files)


def load_jsonl(path: str) -> list[dict]:
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def build_prompt(sentence: str, options: Iterable[str]) -> str:
    options_text = ", ".join(options)
    return PROMPT_TEMPLATE.format(sentence=sentence, options=options_text)


def mask_verb(sentence: str, verb: str) -> str:
    pattern = rf"\b{re.escape(verb)}\b"
    masked, count = re.subn(pattern, "_____", sentence, count=1)
    return masked if count > 0 else sentence


def compute_option_probability(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    next_token_probs: torch.Tensor,
    option: str,
) -> tuple[float, list[int]]:
    token_ids = tokenizer.encode(option, add_special_tokens=False)
    if not token_ids:
        return 0.0, token_ids
    if len(token_ids) == 1:
        return next_token_probs[token_ids[0]].item(), token_ids

    option_ids = torch.tensor([token_ids], device=model.device)
    input_ids = torch.cat([model_inputs["input_ids"], option_ids], dim=1)
    if "attention_mask" in model_inputs:
        attention_mask = torch.cat(
            [
                model_inputs["attention_mask"],
                torch.ones_like(option_ids, device=model.device),
            ],
            dim=1,
        )
    else:
        attention_mask = None
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = model_inputs["input_ids"].shape[-1]

    total_logprob = 0.0
    for idx, tok_id in enumerate(token_ids):
        total_logprob += log_probs[prompt_len + idx - 1, tok_id].item()
    return math.exp(total_logprob), token_ids


def score_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    options: list[str],
    top_k: int,
) -> tuple[dict, list[tuple[str, float, list[int]]], str, list[tuple[str, float]]]:
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**model_inputs)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    option_results: list[tuple[str, float, list[int]]] = []
    for option in options:
        prob, token_ids = compute_option_probability(
            model, tokenizer, model_inputs, probs, option
        )
        option_results.append((option, prob, token_ids))

    best_option = max(option_results, key=lambda item: item[1])[0] if option_results else ""
    top_k = min(top_k, probs.shape[0])
    top_k_vals, top_k_ids = torch.topk(probs, top_k)
    top_k_list = [
        (tokenizer.decode([tok_id]), float(tok_prob))
        for tok_id, tok_prob in zip(top_k_ids.tolist(), top_k_vals.tolist())
    ]
    return model_inputs, option_results, best_option, top_k_list


def generate_from_inputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_inputs: dict,
    max_new_tokens: int,
) -> tuple[str, str]:
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_len = model_inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, input_len:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return completion, full_text


def render_generation_block(
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    full_text: str,
    option_results: list[tuple[str, float, list[int]]],
    best_option: str,
    top_k_list: list[tuple[str, float]],
) -> str:
    lines: list[str] = []
    lines.append("Prompt:\n" + prompt)
    lines.append("\nCompletion:\n" + completion)
    lines.append("\nFull text:\n" + full_text)
    lines.append("\nOption probabilities:")
    for option, prob, token_ids in option_results:
        token_pieces = [tokenizer.decode([tok_id]) for tok_id in token_ids]
        token_info = " ".join(token_pieces) if token_pieces else "(no tokens)"
        lines.append(f"- {option}: {prob:.8f} | tokens: {token_info}")
    if best_option:
        lines.append(f"\nMost likely option: {best_option}")
    lines.append("\n--------------------------------\n")
    lines.append(f"Top {len(top_k_list)} Next-Token Predictions:")
    for rank, (tok_text, tok_prob) in enumerate(top_k_list, start=1):
        lines.append(f"{rank}. '{tok_text}' ({tok_prob:.4f})")
    return "\n".join(lines)


def evaluate_jsonl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    jsonl_path: str,
    output_path: str,
    max_new_tokens: int,
    top_k: int,
    sample_size: int,
    rng: random.Random,
) -> None:
    entries = load_jsonl(jsonl_path)
    total = len(entries)
    if total == 0:
        return

    sample_indices = set(rng.sample(range(total), k=min(sample_size, total)))
    sample_blocks: list[str] = []

    base_correct = 0
    source_correct = 0
    base_ties = 0
    source_ties = 0

    progress = tqdm(entries, desc=os.path.basename(jsonl_path), unit="sent")
    for idx, entry in enumerate(progress):
        base_sentence = entry["base_sentence"]
        source_sentence = entry["source_sentence"]
        mv_base = entry["MV_base"]
        mv_source = entry["MV_source"]

        base_options = [mv_base, mv_source]
        base_sentence_masked = mask_verb(base_sentence, mv_base)
        base_prompt = build_prompt(base_sentence_masked, base_options)
        (
            base_inputs,
            base_option_results,
            base_best_option,
            base_top_k,
        ) = score_prompt(model, tokenizer, base_prompt, base_options, top_k)
        base_prob_map = {opt: prob for opt, prob, _ in base_option_results}
        base_correct_prob = base_prob_map.get(mv_base, 0.0)
        base_wrong_prob = base_prob_map.get(mv_source, 0.0)
        if base_correct_prob > base_wrong_prob:
            base_correct += 1
        elif base_correct_prob == base_wrong_prob:
            base_ties += 1

        source_options = [mv_source, mv_base]
        source_sentence_masked = mask_verb(source_sentence, mv_source)
        source_prompt = build_prompt(source_sentence_masked, source_options)
        (
            source_inputs,
            source_option_results,
            source_best_option,
            source_top_k,
        ) = score_prompt(model, tokenizer, source_prompt, source_options, top_k)
        source_prob_map = {opt: prob for opt, prob, _ in source_option_results}
        source_correct_prob = source_prob_map.get(mv_source, 0.0)
        source_wrong_prob = source_prob_map.get(mv_base, 0.0)
        if source_correct_prob > source_wrong_prob:
            source_correct += 1
        elif source_correct_prob == source_wrong_prob:
            source_ties += 1

        if idx in sample_indices:
            base_completion, base_full = generate_from_inputs(
                model, tokenizer, base_inputs, max_new_tokens
            )
            base_block = render_generation_block(
                tokenizer,
                base_prompt,
                base_completion,
                base_full,
                base_option_results,
                base_best_option,
                base_top_k,
            )
            sample_blocks.append(
                "\n".join(
                    [
                        f"\n=== Sample {len(sample_blocks) + 1} (base) ===",
                        f"Correct: {mv_base} | Wrong: {mv_source}",
                        base_block,
                    ]
                )
            )

            source_completion, source_full = generate_from_inputs(
                model, tokenizer, source_inputs, max_new_tokens
            )
            source_block = render_generation_block(
                tokenizer,
                source_prompt,
                source_completion,
                source_full,
                source_option_results,
                source_best_option,
                source_top_k,
            )
            sample_blocks.append(
                "\n".join(
                    [
                        f"\n=== Sample {len(sample_blocks) + 1} (source) ===",
                        f"Correct: {mv_source} | Wrong: {mv_base}",
                        source_block,
                    ]
                )
            )

    base_accuracy = base_correct / total
    source_accuracy = source_correct / total

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"jsonl_path: {jsonl_path}\n")
        handle.write(f"total_entries: {total}\n")
        handle.write(f"base_overall_accuracy: {base_accuracy:.6f}\n")
        handle.write(f"source_overall_accuracy: {source_accuracy:.6f}\n")
        handle.write(f"base_correct: {base_correct}\n")
        handle.write(f"source_correct: {source_correct}\n")
        handle.write(f"base_ties: {base_ties}\n")
        handle.write(f"source_ties: {source_ties}\n")
        handle.write("\n")
        handle.write("\n".join(sample_blocks))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    jsonl_files = list_jsonl_files(args.data_dir)
    if not jsonl_files:
        raise SystemExit(f"No JSONL files found in {args.data_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for model_name in args.models:
        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
        model.eval()

        model_dir = os.path.join(args.results_dir, sanitize_model_name(model_name))
        for jsonl_path in jsonl_files:
            rel_path = os.path.relpath(jsonl_path, args.data_dir)
            rel_dir = os.path.dirname(rel_path)
            base_name = os.path.splitext(os.path.basename(jsonl_path))[0] + ".txt"
            output_path = os.path.join(model_dir, rel_dir, base_name)
            print(f"Evaluating {jsonl_path}")
            evaluate_jsonl(
                model,
                tokenizer,
                jsonl_path,
                output_path,
                args.max_new_tokens,
                args.top_k,
                args.sample_size,
                rng,
            )

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
