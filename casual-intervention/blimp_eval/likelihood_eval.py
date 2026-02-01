import argparse
import os
from typing import Dict, Iterable, List

import torch
import yaml
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "nyu-mll/blimp"


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


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


def encode_sentence(tokenizer: AutoTokenizer, sentence: str) -> List[int]:
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        if not ids or ids[0] != tokenizer.bos_token_id:
            ids = [tokenizer.bos_token_id] + ids
    return ids


def iter_batches(dataset: Iterable[Dict], batch_size: int) -> Iterable[List[Dict]]:
    batch: List[Dict] = []
    for example in dataset:
        batch.append(example)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_padded_batch(
    token_ids_list: List[List[int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    max_len = max(len(ids) for ids in token_ids_list)
    input_ids = []
    attention_mask = []
    seq_lens = []
    for ids in token_ids_list:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        seq_lens.append(len(ids))
    return (
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        seq_lens,
    )


def sentence_logprobs_batch(
    model: AutoModelForCausalLM,
    token_ids_list: List[List[int]],
    pad_id: int,
) -> List[float]:
    input_ids, attention_mask, seq_lens = build_padded_batch(token_ids_list, pad_id)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    logprobs = []
    for batch_idx, seq_len in enumerate(seq_lens):
        total = 0.0
        for pos in range(1, seq_len):
            token_id = input_ids[batch_idx, pos].item()
            total += log_probs[batch_idx, pos - 1, token_id].item()
        logprobs.append(total)
    return logprobs


def compute_component_stats(
    dataset,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch_size: int,
    pad_id: int,
) -> Dict[str, float | int]:
    total_pairs = 0
    correct_pairs = 0
    sum_logprob_diff = 0.0

    for batch in iter_batches(dataset, batch_size):
        good_tokens = [encode_sentence(tokenizer, ex["sentence_good"]) for ex in batch]
        bad_tokens = [encode_sentence(tokenizer, ex["sentence_bad"]) for ex in batch]

        good_logprobs = sentence_logprobs_batch(model, good_tokens, pad_id)
        bad_logprobs = sentence_logprobs_batch(model, bad_tokens, pad_id)

        for lp_good, lp_bad in zip(good_logprobs, bad_logprobs):
            total_pairs += 1
            sum_logprob_diff += lp_good - lp_bad
            if lp_good > lp_bad:
                correct_pairs += 1

    wrong_pairs = total_pairs - correct_pairs
    accuracy = correct_pairs / total_pairs if total_pairs else 0.0
    avg_logprob_diff = sum_logprob_diff / total_pairs if total_pairs else 0.0
    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "wrong_pairs": wrong_pairs,
        "accuracy": accuracy,
        "avg_logprob_diff": avg_logprob_diff,
    }


def write_report(report_path: str, stats_by_component: Dict[str, Dict]) -> None:
    total_pairs = sum(stats["total_pairs"] for stats in stats_by_component.values())
    total_correct = sum(stats["correct_pairs"] for stats in stats_by_component.values())
    total_wrong = sum(stats["wrong_pairs"] for stats in stats_by_component.values())
    overall_accuracy = total_correct / total_pairs if total_pairs else 0.0
    avg_diff = (
        sum(
            stats["avg_logprob_diff"] * stats["total_pairs"]
            for stats in stats_by_component.values()
        )
        / total_pairs
        if total_pairs
        else 0.0
    )

    lines = []
    lines.append("BLiMP sentence likelihood report")
    lines.append("")
    lines.append("Overall:")
    lines.append(f"  total_pairs: {total_pairs}")
    lines.append(f"  correct_pairs: {total_correct}")
    lines.append(f"  wrong_pairs: {total_wrong}")
    lines.append(f"  accuracy: {overall_accuracy:.4f}")
    lines.append(f"  avg_logprob_diff: {avg_diff:.4f}")
    lines.append("")
    lines.append("Per component:")

    for component, stats in stats_by_component.items():
        lines.append(f"- {component}")
        lines.append(f"  total_pairs: {stats['total_pairs']}")
        lines.append(f"  correct_pairs: {stats['correct_pairs']}")
        lines.append(f"  wrong_pairs: {stats['wrong_pairs']}")
        lines.append(f"  accuracy: {stats['accuracy']:.4f}")
        lines.append(f"  avg_logprob_diff: {stats['avg_logprob_diff']:.4f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BLiMP accuracy using sentence log-likelihoods."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_data = load_yaml_config(args.config)

    dataset_name = config_data.get("dataset_name", DATASET_NAME)
    configs = config_data.get("configs") or get_dataset_config_names(dataset_name)
    output_dir = config_data.get("output_dir", "output")
    batch_size = int(config_data.get("batch_size", 8))
    device_map = resolve_device_map(config_data.get("device_map"))
    torch_dtype = resolve_torch_dtype(config_data.get("torch_dtype"))
    models_cfg = config_data.get("models")

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    models = normalize_models(models_cfg, args.models)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(base_dir, "output", "reports")
    os.makedirs(report_dir, exist_ok=True)

    for model in models:
        tokenizer, model_obj = load_model(
            model["path"], hf_token, device_map, torch_dtype
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        stats_by_component: Dict[str, Dict] = {}
        for config in configs:
            dataset = load_dataset(dataset_name, config, split="train")
            stats_by_component[config] = compute_component_stats(
                dataset, tokenizer, model_obj, batch_size, pad_id
            )

        report_path = os.path.join(
            report_dir, f"blimp_likelihood_{model['name']}.txt"
        )
        write_report(report_path, stats_by_component)


if __name__ == "__main__":
    main()
