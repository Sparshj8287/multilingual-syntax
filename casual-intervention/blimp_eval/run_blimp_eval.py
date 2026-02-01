import argparse
import json
import math
import os
import random
from typing import Dict, Iterable, List, Tuple

import torch
import yaml
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "nyu-mll/blimp"
PROMPT_TEMPLATE = (
    "Given sentences A and B, choose the one which is grammatically acceptable. "
    'Give your answer as "A" or "B" only.\n'
    "Sentence A: {sentence_a}\n"
    "Sentence B: {sentence_b}\n"
    "Answer:"
)
CANDIDATES = {"A": " A", "B": " B"}


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def encode_prompt(tokenizer: AutoTokenizer, prompt: str) -> List[int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        if not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id:
            prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    return prompt_ids


def encode_candidate(tokenizer: AutoTokenizer, candidate: str) -> List[int]:
    return tokenizer.encode(candidate, add_special_tokens=False)


def build_padded_batch(
    prompt_ids_list: List[List[int]],
    candidate_ids: List[int],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    sequences = [prompt_ids + candidate_ids for prompt_ids in prompt_ids_list]
    max_len = max(len(seq) for seq in sequences)
    input_ids = []
    attention_mask = []
    prompt_lens = []
    for prompt_ids, seq in zip(prompt_ids_list, sequences):
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
        prompt_lens.append(len(prompt_ids))
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        prompt_lens,
    )


def score_candidate_batch(
    model: AutoModelForCausalLM,
    prompt_ids_list: List[List[int]],
    candidate_ids: List[int],
    pad_id: int,
) -> List[float]:
    input_ids, attention_mask, prompt_lens = build_padded_batch(
        prompt_ids_list, candidate_ids, pad_id
    )
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    logprobs = []
    for batch_idx, prompt_len in enumerate(prompt_lens):
        start = prompt_len - 1
        total_logprob = 0.0
        for i, token_id in enumerate(candidate_ids):
            pos = start + i
            total_logprob += log_probs[batch_idx, pos, token_id].item()
        logprobs.append(total_logprob)
    return logprobs


def normalize_probs(logprob_a: float, logprob_b: float) -> Tuple[float, float]:
    max_lp = max(logprob_a, logprob_b)
    denom = math.exp(logprob_a - max_lp) + math.exp(logprob_b - max_lp)
    prob_a = math.exp(logprob_a - max_lp) / denom
    prob_b = math.exp(logprob_b - max_lp) / denom
    return prob_a, prob_b


def score_pairs_batch(
    sentence_a_list: List[str],
    sentence_b_list: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt_template: str,
    cand_a_ids: List[int],
    cand_b_ids: List[int],
    pad_id: int,
) -> List[Dict[str, float]]:
    prompts = [
        prompt_template.format(sentence_a=a, sentence_b=b)
        for a, b in zip(sentence_a_list, sentence_b_list)
    ]
    prompt_ids_list = [encode_prompt(tokenizer, prompt) for prompt in prompts]
    logprob_a = score_candidate_batch(model, prompt_ids_list, cand_a_ids, pad_id)
    logprob_b = score_candidate_batch(model, prompt_ids_list, cand_b_ids, pad_id)
    scored = []
    for lp_a, lp_b in zip(logprob_a, logprob_b):
        prob_a, prob_b = normalize_probs(lp_a, lp_b)
        pred = "A" if prob_a >= prob_b else "B"
        scored.append(
            {
                "pred": pred,
                "prob_a": prob_a,
                "prob_b": prob_b,
            }
        )
    return scored


def load_model(
    model_path: str,
    hf_token: str | None,
    device_map: str | None,
    torch_dtype: str | None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype or "auto",
        device_map=device_map or "auto",
        token=hf_token,
    )
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def iter_batches(dataset: Iterable[Dict], batch_size: int) -> Iterable[List[Dict]]:
    batch: List[Dict] = []
    for example in dataset:
        batch.append(example)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_for_model(
    model_name: str,
    model_path: str,
    configs: List[str],
    output_dir: str,
    max_examples: int | None,
    batch_size: int,
    prompt_template: str,
    dataset_name: str,
    device_map: str | None,
    torch_dtype: str | None,
    swap_fraction: float,
    swap_seed: int | None,
    hf_token: str | None,
) -> None:
    tokenizer, model = load_model(model_path, hf_token, device_map, torch_dtype)
    model_dir = os.path.join(output_dir, sanitize_model_id(model_name))
    os.makedirs(model_dir, exist_ok=True)

    cand_a_ids = encode_candidate(tokenizer, CANDIDATES["A"])
    cand_b_ids = encode_candidate(tokenizer, CANDIDATES["B"])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for config in configs:
        rng = random.Random(swap_seed if swap_seed is not None else None)
        dataset = load_dataset(dataset_name, config, split="train")
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        output_path = os.path.join(model_dir, f"en_blimp_{config}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for batch in iter_batches(dataset, batch_size):
                sentence_a_list = []
                sentence_b_list = []
                pair_gold_list = []
                pair_swapped_list = []
                for ex in batch:
                    do_swap = swap_fraction > 0 and rng.random() < swap_fraction
                    if do_swap:
                        sentence_a_list.append(ex["sentence_bad"])
                        sentence_b_list.append(ex["sentence_good"])
                        pair_gold_list.append("B")
                        pair_swapped_list.append(True)
                    else:
                        sentence_a_list.append(ex["sentence_good"])
                        sentence_b_list.append(ex["sentence_bad"])
                        pair_gold_list.append("A")
                        pair_swapped_list.append(False)

                scored_pairs = score_pairs_batch(
                    sentence_a_list,
                    sentence_b_list,
                    tokenizer,
                    model,
                    prompt_template,
                    cand_a_ids,
                    cand_b_ids,
                    pad_id,
                )

                for example, pair_score, pair_gold, pair_swapped in zip(
                    batch, scored_pairs, pair_gold_list, pair_swapped_list
                ):
                    record = dict(example)
                    record.update(
                        {
                            "model_name": model_name,
                            "model_path": model_path,
                            "prompt_template": prompt_template,
                            "pair_gold": pair_gold,
                            "pair_swapped": pair_swapped,
                            "pair_pred": pair_score["pred"],
                            "pair_prob_A": pair_score["prob_a"],
                            "pair_prob_B": pair_score["prob_b"],
                        }
                    )
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")



def normalize_models(models_cfg: List | None, models_cli: List[str]) -> List[Dict[str, str]]:
    if not models_cfg:
        return [{"name": model_id, "path": model_id} for model_id in models_cli]

    normalized = []
    for entry in models_cfg:
        if isinstance(entry, str):
            name = entry
            path = entry
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("id") or entry.get("path")
            path = entry.get("path") or entry.get("id") or entry.get("name")
        else:
            raise ValueError(f"Unsupported model entry: {entry}")
        if not name or not path:
            raise ValueError(f"Model entry must include name/path: {entry}")
        normalized.append({"name": name, "path": path})
    return normalized


def load_yaml_config(path: str | None) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_torch_dtype(dtype_value: str | None):
    if not dtype_value:
        return None
    if isinstance(dtype_value, str):
        if dtype_value == "auto":
            return "auto"
        if hasattr(torch, dtype_value):
            return getattr(torch, dtype_value)
    return dtype_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BLiMP probability-based good/bad scoring for causal LMs."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config for evaluation settings.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["google/gemma-3-1b", "meta-llama/Llama-3.2-1B"],
        help="Model ids to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Base directory for model outputs.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional subset of BLiMP configs to run.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on examples per config (debug).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sentence pairs per batch.",
    )
    parser.add_argument(
        "--prompt_template",
        default=PROMPT_TEMPLATE,
        help="Prompt template with {sentence_a} and {sentence_b} placeholders.",
    )
    parser.add_argument(
        "--dataset_name",
        default=DATASET_NAME,
        help="Dataset name on Hugging Face.",
    )
    parser.add_argument(
        "--swap_fraction",
        type=float,
        default=0.0,
        help="Fraction of rows to swap sentence A/B (0.0 to 1.0).",
    )
    parser.add_argument(
        "--swap_seed",
        type=int,
        default=None,
        help="Random seed for sentence swapping.",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Override transformers device_map (default: auto).",
    )
    parser.add_argument(
        "--torch_dtype",
        default=None,
        help="Override torch_dtype (e.g. float16, bfloat16).",
    )
    return parser.parse_args()


def resolve_device_map(device_map: str | None) -> str | Dict | None:
    if not device_map:
        return "auto"
    if device_map == "gpu0":
        return "cuda:0"
    if device_map == "gpu1":
        return "cuda:1"
    if device_map == "gpu2":
        return "cuda:2"
    return device_map


def main() -> None:
    args = parse_args()
    config_data = load_yaml_config(args.config)

    dataset_name = config_data.get("dataset_name", args.dataset_name)
    configs = config_data.get("configs", args.configs) or get_dataset_config_names(
        dataset_name
    )
    output_dir = config_data.get("output_dir", args.output_dir)
    max_examples = config_data.get("max_examples", args.max_examples)
    batch_size = config_data.get("batch_size", args.batch_size)
    prompt_template = config_data.get("prompt_template", args.prompt_template)
    device_map = resolve_device_map(config_data.get("device_map", args.device_map))
    torch_dtype = resolve_torch_dtype(config_data.get("torch_dtype", args.torch_dtype))
    swap_fraction = config_data.get("swap_fraction", args.swap_fraction)
    swap_seed = config_data.get("swap_seed", args.swap_seed)
    models_cfg = config_data.get("models")

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    models = normalize_models(models_cfg, args.models)

    for model in models:
        run_for_model(
            model["name"],
            model["path"],
            configs,
            output_dir,
            max_examples,
            batch_size,
            prompt_template,
            dataset_name,
            device_map,
            torch_dtype,
            swap_fraction,
            swap_seed,
            hf_token,
        )


if __name__ == "__main__":
    main()
