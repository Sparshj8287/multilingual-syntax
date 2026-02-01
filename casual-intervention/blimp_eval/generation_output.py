import argparse
import os
from typing import Dict, List

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
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
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


def normalize_probs(logprob_a: float, logprob_b: float) -> tuple[float, float]:
    max_lp = max(logprob_a, logprob_b)
    denom = torch.exp(torch.tensor(logprob_a - max_lp)) + torch.exp(
        torch.tensor(logprob_b - max_lp)
    )
    prob_a = float(torch.exp(torch.tensor(logprob_a - max_lp)) / denom)
    prob_b = float(torch.exp(torch.tensor(logprob_b - max_lp)) / denom)
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


def generate_first_tokens(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    max_new_tokens: int,
    pad_id: int,
) -> str:
    prompt_ids = encode_prompt(tokenizer, prompt)
    input_ids = torch.tensor([prompt_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    generated_ids = output_ids[0][len(prompt_ids) : len(prompt_ids) + max_new_tokens]
    tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    return " ".join(tokens)


def load_model(
    model_path: str,
    hf_token: str | None,
    device_map: str | None,
    torch_dtype: str | None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
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


def resolve_torch_dtype(dtype_value: str | None):
    if not dtype_value:
        return None
    if isinstance(dtype_value, str):
        if dtype_value == "auto":
            return "auto"
        if hasattr(torch, dtype_value):
            return getattr(torch, dtype_value)
    return dtype_value


def load_yaml_config(path: str | None) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write generation output for the first BLiMP config."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config for evaluation settings.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Model ids to evaluate (overridden by config models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_data = load_yaml_config(args.config)

    dataset_name = config_data.get("dataset_name", DATASET_NAME)
    configs = config_data.get("configs") or get_dataset_config_names(dataset_name)
    output_dir = config_data.get("output_dir", "output")
    prompt_template = config_data.get("prompt_template", PROMPT_TEMPLATE)
    device_map = resolve_device_map(config_data.get("device_map"))
    torch_dtype = resolve_torch_dtype(config_data.get("torch_dtype"))
    generation_output = config_data.get("generation_output", False)
    generation_num_sentences = int(config_data.get("generation_num_sentences", 20))
    generation_max_tokens = int(config_data.get("generation_max_tokens", 10))
    models_cfg = config_data.get("models")

    if not generation_output:
        raise SystemExit(
            "generation_output is disabled in config. Set generation_output: true."
        )

    if not configs:
        raise SystemExit("No dataset configs available.")
    first_config = configs[0]
    dataset = load_dataset(dataset_name, first_config, split="train")

    models = normalize_models(models_cfg, args.models)
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for model in models:
        tokenizer, model_obj = load_model(
            model["path"], hf_token, device_map, torch_dtype
        )
        cand_a_ids = encode_candidate(tokenizer, CANDIDATES["A"])
        cand_b_ids = encode_candidate(tokenizer, CANDIDATES["B"])
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        total_needed = generation_num_sentences * 2
        if len(dataset) < total_needed:
            raise SystemExit(
                f"Not enough rows: need {total_needed}, got {len(dataset)}"
            )

        pairs = []
        for idx in range(generation_num_sentences):
            ex = dataset[int(idx)]
            pairs.append(
                {
                    "gold": "A",
                    "sentence_a": ex["sentence_good"],
                    "sentence_b": ex["sentence_bad"],
                }
            )
        for idx in range(generation_num_sentences, total_needed):
            ex = dataset[int(idx)]
            pairs.append(
                {
                    "gold": "B",
                    "sentence_a": ex["sentence_bad"],
                    "sentence_b": ex["sentence_good"],
                }
            )

        for start in range(0, len(pairs), max(1, generation_num_sentences)):
            batch = pairs[start : start + max(1, generation_num_sentences)]
            sentence_a_list = [p["sentence_a"] for p in batch]
            sentence_b_list = [p["sentence_b"] for p in batch]
            scored = score_pairs_batch(
                sentence_a_list,
                sentence_b_list,
                tokenizer,
                model_obj,
                prompt_template,
                cand_a_ids,
                cand_b_ids,
                pad_id,
            )
            for pair, score in zip(batch, scored):
                pair["pred"] = score["pred"]

        generation_dir = os.path.join(output_dir, "generation_output")
        os.makedirs(generation_dir, exist_ok=True)
        report_path = os.path.join(
            generation_dir, f"blimp_generation_{model['name']}.txt"
        )
        lines = []
        for pair in pairs:
            prompt = prompt_template.format(
                sentence_a=pair["sentence_a"], sentence_b=pair["sentence_b"]
            )
            generated = generate_first_tokens(
                tokenizer, model_obj, prompt, generation_max_tokens, pad_id
            )
            lines.append(f"{pair['gold']},{pair['pred']},{generated}")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
