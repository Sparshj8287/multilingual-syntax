#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

SCRIPT_DIR = Path(__file__).resolve().parent
CASUAL_INTERVENTION_ROOT = SCRIPT_DIR.parent.parent
PYVENE_ROOT = CASUAL_INTERVENTION_ROOT / "pyvene"
if str(PYVENE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYVENE_ROOT))

from pyvene import (  # noqa: E402
    BoundlessRotatedSpaceIntervention,
    IntervenableConfig,
    IntervenableModel,
    RepresentationConfig,
    count_parameters,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Boundless DAS on simple-agreement JSONL data."
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_path(base_dir: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def sanitize_model_dir_name(model_name: str, model_path: str) -> str:
    candidate = (model_name or "").strip()
    if not candidate:
        candidate = model_path.strip()
    if "/" in candidate:
        candidate = candidate.split("/")[-1]
    return candidate.replace(" ", "_")


def resolve_device_map(device_map_value: Any):
    if device_map_value is None:
        return None
    if isinstance(device_map_value, str):
        normalized = device_map_value.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized == "gpu0":
            return "cuda:0"
        if normalized == "gpu1":
            return "cuda:1"
        if normalized == "gpu2":
            return "cuda:2"
    return device_map_value


def resolve_torch_dtype(dtype_value: str | None):
    if not dtype_value:
        return None
    normalized = dtype_value.strip().lower()
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(
            "Unsupported torch_dtype. Use one of: auto, float32/fp32, float16/fp16, bfloat16/bf16."
        )
    return mapping[normalized]


def load_model_and_tokenizer(
    model_path: str,
    runtime_cfg: dict[str, Any],
    hf_token: str | None,
):
    use_fast = bool(runtime_cfg.get("use_fast_tokenizer", True))
    device = torch.device(runtime_cfg.get("device", "cuda"))
    device_map = resolve_device_map(runtime_cfg.get("device_map"))
    torch_dtype = resolve_torch_dtype(runtime_cfg.get("torch_dtype", "bfloat16"))

    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
    if not torch.cuda.is_available() and device_map is not None:
        device_map = {"": "cpu"}
    if device.type == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
        use_fast=use_fast,
    )

    if tokenizer.pad_token_id is None:

        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype if torch_dtype is not None else "auto",
        device_map=device_map,
        token=hf_token,
    )

    print("Loaded model name:", model.name_or_path)
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if device_map is None:
        model.to(device)
    model.eval()
    return tokenizer, model, device, device_map


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def context_before_final_verb(sentence: str, final_verb: str) -> str:
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


def normalize_prefix(prefix: str) -> str:
    value = str(prefix)
    if not value.endswith(" "):
        value = value + " "
    return value


def strict_single_token_id(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    row_idx: int,
    field_name: str,
) -> int:
    token_ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected single-token verb for {field_name} at row {row_idx}, got token ids {token_ids} for '{text}'."
        )
    return int(token_ids[0])


def tokenize_prefix(tokenizer: AutoTokenizer, prefix: str) -> list[int]:
    token_ids = tokenizer.encode(prefix, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        if not token_ids or token_ids[0] != bos_id:
            token_ids = [bos_id] + token_ids
    if not token_ids:
        raise ValueError(f"Prefix tokenized to empty ids: {repr(prefix)}")
    return token_ids


def build_examples(
    rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        mv_base = str(row.get("MV_base", "")).strip()
        mv_source = str(row.get("MV_source", "")).strip()
        if not mv_base or not mv_source:
            raise ValueError(f"Missing MV_base/MV_source in row {idx}")

        base_prefix = row.get("base_prefix")
        if base_prefix is None:
            base_sentence = row.get("base_sentence")
            if base_sentence is None:
                raise ValueError(
                    f"Row {idx} missing both base_prefix and base_sentence fields."
                )
            base_prefix = context_before_final_verb(str(base_sentence), mv_base)

        source_prefix = row.get("source_prefix")
        if source_prefix is None:
            source_sentence = row.get("source_sentence")
            if source_sentence is None:
                raise ValueError(
                    f"Row {idx} missing both source_prefix and source_sentence fields."
                )
            source_prefix = context_before_final_verb(str(source_sentence), mv_source)

        base_prefix = normalize_prefix(str(base_prefix))
        source_prefix = normalize_prefix(str(source_prefix))

        id_base = strict_single_token_id(
            tokenizer,
            mv_base,
            row_idx=idx,
            field_name="MV_base",
        )
        id_source = strict_single_token_id(
            tokenizer,
            mv_source,
            row_idx=idx,
            field_name="MV_source",
        )

        examples.append(
            {
                "row_idx": idx,
                "base_prefix": base_prefix,
                "source_prefix": source_prefix,
                "mv_base": mv_base,
                "mv_source": mv_source,
                "id_base": id_base,
                "id_source": id_source,
                "base_prefix_ids": tokenize_prefix(tokenizer, base_prefix),
                "source_prefix_ids": tokenize_prefix(tokenizer, source_prefix),
            }
        )
    return examples


def split_examples(
    examples: list[dict[str, Any]],
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
):
    required = train_size + val_size + test_size
    if len(examples) < required:
        raise ValueError(
            f"Dataset too small: {len(examples)} rows, but {required} required for train/val/test."
        )
    shuffled = list(examples)
    rng = random.Random(seed)
    set_seed(seed)
    rng.shuffle(shuffled)
    train_split = shuffled[:train_size]
    val_split = shuffled[train_size : train_size + val_size]
    test_split = shuffled[train_size + val_size : required]
    return train_split, val_split, test_split


def split_train_val_examples(
    train_examples: list[dict[str, Any]],
    *,
    val_size: int,
    seed: int,
    train_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size <= 0:
        raise ValueError("dataset.val_size must be > 0.")
    shuffled = list(train_examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if train_size is not None and train_size > 0:
        required = train_size + val_size
        if len(shuffled) < required:
            raise ValueError(
                "Training file too small for requested train/val split: "
                f"{len(shuffled)} < {required}"
            )
        train_split = shuffled[:train_size]
        val_split = shuffled[train_size : train_size + val_size]
        return train_split, val_split

    if len(shuffled) <= val_size:
        raise ValueError(
            "Training file too small for validation split: "
            f"{len(shuffled)} <= {val_size}"
        )
    val_split = shuffled[:val_size]
    train_split = shuffled[val_size:]
    return train_split, val_split


def normalize_nua_values(raw_value: Any) -> list[int]:
    if raw_value is None:
        return [1]
    if isinstance(raw_value, (int, float)):
        return [int(raw_value)]
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return [1]
        if "," in value:
            return [int(v.strip()) for v in value.split(",") if v.strip()]
        return [int(value)]
    if isinstance(raw_value, (list, tuple)):
        values = [int(v) for v in raw_value]
        if not values:
            raise ValueError("dataset.NUA list cannot be empty.")
        return values
    raise ValueError(f"Unsupported dataset.NUA value: {raw_value}")


def get_auto_halved_batch_size(
    base_batch_size: int,
    nua: int,
    *,
    min_batch_size: int = 1,
) -> int:
    """Halve batch size as attractor count (NUA) increases."""
    if base_batch_size <= 0:
        raise ValueError("training.batch_size and training.eval_batch_size must be > 0.")
    if nua <= 0:
        raise ValueError("NUA values must be positive integers.")
    halving_steps = max(0, nua - 1)
    return max(min_batch_size, base_batch_size // (2 ** halving_steps))


def build_dataset_run_specs(
    dataset_cfg: dict[str, Any],
    config_dir: Path,
) -> list[dict[str, Any]]:
    dataset_name = str(dataset_cfg.get("dataset_name", "")).strip()
    variation = str(dataset_cfg.get("variation", "")).strip()
    if not dataset_name or not variation:
        raise ValueError(
            "dataset.dataset_name and dataset.variation are required."
        )

    data_root = resolve_path(
        config_dir,
        str(
            dataset_cfg.get(
                "data_root", "../data/data_generators/templates/data"
            )
        ),
    )
    train_split_dir = str(dataset_cfg.get("train_split_dir", "train")).strip()
    test_split_dir = str(dataset_cfg.get("test_split_dir", "test")).strip()
    file_name_template = str(
        dataset_cfg.get(
            "file_name_template",
            "{dataset_name}_pairs_{variation}_{nua}.jsonl",
        )
    )
    nua_values = normalize_nua_values(
        dataset_cfg.get("NUA", dataset_cfg.get("nua"))
    )

    runs: list[dict[str, Any]] = []
    for nua in nua_values:
        file_name = file_name_template.format(
            dataset_name=dataset_name,
            variation=variation,
            nua=nua,
        )
        train_path = (
            data_root
            / dataset_name
            / train_split_dir
            / variation
            / file_name
        )
        test_path = (
            data_root
            / dataset_name
            / test_split_dir
            / variation
            / file_name
        )
        if not train_path.exists():
            raise ValueError(f"Training file does not exist: {train_path}")
        if not test_path.exists():
            raise ValueError(f"Test file does not exist: {test_path}")
        runs.append(
            {
                "dataset_name": dataset_name,
                "variation": variation,
                "nua": int(nua),
                "train_path": train_path,
                "test_path": test_path,
            }
        )
    return runs


def collate_examples(
    examples: list[dict[str, Any]],
    *,
    pad_id: int,
    intervene_direction: str,
) -> dict[str, Any]:
    main_ids_list: list[list[int]] = []
    source_ids_list: list[list[int]] = []
    target_ids: list[int] = []
    wrong_ids: list[int] = []
    target_words: list[str] = []
    wrong_words: list[str] = []
    main_prefixes: list[str] = []
    source_prefixes: list[str] = []
    row_indices: list[int] = []

    for ex in examples:
        if intervene_direction == "source_to_base":
            main_ids = ex["base_prefix_ids"]
            source_ids = ex["source_prefix_ids"]
            target_id = ex["id_source"]
            wrong_id = ex["id_base"]
            target_word = ex["mv_source"]
            wrong_word = ex["mv_base"]
            main_prefix = ex["base_prefix"]
            source_prefix = ex["source_prefix"]
        elif intervene_direction == "base_to_source":
            main_ids = ex["source_prefix_ids"]
            source_ids = ex["base_prefix_ids"]
            target_id = ex["id_base"]
            wrong_id = ex["id_source"]
            target_word = ex["mv_base"]
            wrong_word = ex["mv_source"]
            main_prefix = ex["source_prefix"]
            source_prefix = ex["base_prefix"]
        else:
            raise ValueError(f"Unsupported intervene_direction: {intervene_direction}")

        main_ids_list.append(main_ids)
        source_ids_list.append(source_ids)
        target_ids.append(target_id)
        wrong_ids.append(wrong_id)
        target_words.append(target_word)
        wrong_words.append(wrong_word)
        main_prefixes.append(main_prefix)
        source_prefixes.append(source_prefix)
        row_indices.append(int(ex["row_idx"]))

    max_len = max(
        max(len(ids) for ids in main_ids_list),
        max(len(ids) for ids in source_ids_list),
    )
    batch_size = len(examples)

    main_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    source_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    main_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    source_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    # Gemma-3 requires token_type_ids in train-mode forward; text-only inputs are all zeros.
    main_token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    source_token_type_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    main_last_positions: list[int] = []
    source_last_positions: list[int] = []

    for i, (main_ids, source_ids) in enumerate(zip(main_ids_list, source_ids_list)):
        main_len = len(main_ids)
        source_len = len(source_ids)
        main_input_ids[i, :main_len] = torch.tensor(main_ids, dtype=torch.long)
        source_input_ids[i, :source_len] = torch.tensor(source_ids, dtype=torch.long)
        main_attention_mask[i, :main_len] = 1
        source_attention_mask[i, :source_len] = 1
        main_last_positions.append(main_len - 1)
        source_last_positions.append(source_len - 1)

    main_positions = [[pos] for pos in main_last_positions]
    source_positions = [[pos] for pos in source_last_positions]

    return {
        "main_input_ids": main_input_ids,
        "main_attention_mask": main_attention_mask,
        "main_token_type_ids": main_token_type_ids,
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "source_token_type_ids": source_token_type_ids,
        "main_last_positions": torch.tensor(main_last_positions, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "wrong_ids": torch.tensor(wrong_ids, dtype=torch.long),
        "main_positions": main_positions,
        "source_positions": source_positions,
        "target_words": target_words,
        "wrong_words": wrong_words,
        "main_prefixes": main_prefixes,
        "source_prefixes": source_prefixes,
        "row_indices": row_indices,
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_intervened_pair_logits(
    intervenable: IntervenableModel,
    batch: dict[str, Any],
) -> torch.Tensor:
    unit_locations = {
        "sources->base": ([batch["source_positions"]], [batch["main_positions"]])
    }
    _, counterfactual_outputs = intervenable(
        {
            "input_ids": batch["main_input_ids"],
            "attention_mask": batch["main_attention_mask"],
            "token_type_ids": batch["main_token_type_ids"],
        },
        [
            {
                "input_ids": batch["source_input_ids"],
                "attention_mask": batch["source_attention_mask"],
                "token_type_ids": batch["source_token_type_ids"],
            }
        ],
        unit_locations,
    )

    logits = counterfactual_outputs.logits
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    last_token_logits = logits[batch_indices, batch["main_last_positions"], :]
    target_logits = torch.gather(
        last_token_logits, dim=1, index=batch["target_ids"].unsqueeze(1)
    ).squeeze(1)
    wrong_logits = torch.gather(
        last_token_logits, dim=1, index=batch["wrong_ids"].unsqueeze(1)
    ).squeeze(1)
    pair_logits = torch.stack([target_logits, wrong_logits], dim=1)
    return pair_logits


def compute_targeted_loss(
    intervenable: IntervenableModel,
    pair_logits: torch.Tensor,
    boundary_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.zeros(pair_logits.shape[0], dtype=torch.long, device=pair_logits.device)
    ce_loss = CrossEntropyLoss()(pair_logits, labels)
    boundary_penalty = torch.zeros((), device=pair_logits.device)
    for _, intervention in intervenable.interventions.items():
        boundary_penalty = boundary_penalty + intervention.intervention_boundaries.sum()
    total_loss = ce_loss + boundary_loss_weight * boundary_penalty
    return total_loss, ce_loss, boundary_penalty


def evaluate(
    intervenable: IntervenableModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    total = 0
    correct = 0
    ties = 0
    loss_sum = 0.0
    margin_sum = 0.0

    intervenable.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            batch = move_batch_to_device(batch, device)
            pair_logits = forward_intervened_pair_logits(intervenable, batch)
            labels = torch.zeros(pair_logits.shape[0], dtype=torch.long, device=device)
            ce_loss = CrossEntropyLoss()(pair_logits, labels)
            margins = pair_logits[:, 0] - pair_logits[:, 1]
            batch_correct = (margins > 0).sum().item()
            batch_ties = (margins == 0).sum().item()

            batch_size = pair_logits.shape[0]
            total += batch_size
            correct += batch_correct
            ties += batch_ties
            margin_sum += margins.sum().item()
            loss_sum += ce_loss.item() * batch_size

    accuracy = correct / total if total else 0.0
    avg_loss = loss_sum / total if total else 0.0
    avg_margin = margin_sum / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "ties": ties,
        "accuracy": accuracy,
        "avg_target_minus_wrong_logit": avg_margin,
        "avg_targeted_ce_loss": avg_loss,
    }


def get_intervention_state(intervenable: IntervenableModel) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    for intervention_key, intervention in intervenable.interventions.items():
        state.append(
            {
                "intervention_key": str(intervention_key),
                "rotate_layer_state_dict": copy.deepcopy(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in intervention.rotate_layer.state_dict().items()
                    }
                ),
                "intervention_boundaries": (
                    intervention.intervention_boundaries.detach().cpu().clone()
                ),
            }
        )
    return state


def load_intervention_state(
    intervenable: IntervenableModel,
    state: list[dict[str, Any]],
) -> None:
    interventions = list(intervenable.interventions.items())
    if len(state) != len(interventions):
        raise ValueError(
            "Checkpoint intervention count does not match current intervenable model: "
            f"{len(state)} != {len(interventions)}"
        )

    for state_item, (_, intervention) in zip(state, interventions):
        intervention.rotate_layer.load_state_dict(state_item["rotate_layer_state_dict"])
        boundary = state_item.get("intervention_boundaries")
        if boundary is not None:
            intervention.intervention_boundaries.data.copy_(
                boundary.to(intervention.intervention_boundaries.device)
            )


def normalize_nested_config(raw_value: Any, *, enabled_key: str = "enabled") -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if raw_value is None:
        return {}
    return {enabled_key: bool(raw_value)}


def build_sample_logs(
    intervenable: IntervenableModel,
    sample_examples: list[dict[str, Any]],
    *,
    pad_id: int,
    intervene_direction: str,
    device: torch.device,
    header: str,
    expected_count: int | None = None,
) -> str:
    if expected_count is not None and len(sample_examples) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} samples for detailed logging, got {len(sample_examples)}."
        )

    lines = [header, ""]
    intervenable.model.eval()
    with torch.no_grad():
        for idx, ex in enumerate(sample_examples, start=1):
            batch = collate_examples(
                [ex],
                pad_id=pad_id,
                intervene_direction=intervene_direction,
            )
            batch = move_batch_to_device(batch, device)
            pair_logits = forward_intervened_pair_logits(intervenable, batch)
            pair_probs = F.softmax(pair_logits[0], dim=-1)

            target_logit = pair_logits[0, 0].item()
            wrong_logit = pair_logits[0, 1].item()
            target_prob = pair_probs[0].item()
            wrong_prob = pair_probs[1].item()
            is_correct = target_logit > wrong_logit

            target_word = batch["target_words"][0]
            wrong_word = batch["wrong_words"][0]
            target_id = int(batch["target_ids"][0].item())
            wrong_id = int(batch["wrong_ids"][0].item())

            lines.extend(
                [
                    f"=== Sample {idx} ===",
                    f"row_idx: {batch['row_indices'][0]}",
                    f"main_prefix: {batch['main_prefixes'][0]}",
                    f"source_prefix: {batch['source_prefixes'][0]}",
                    f"target_word: {target_word} (id={target_id})",
                    f"wrong_word: {wrong_word} (id={wrong_id})",
                    f"raw_target_logit: {target_logit:.8f}",
                    f"raw_wrong_logit: {wrong_logit:.8f}",
                    f"renorm_target_prob: {target_prob:.8f}",
                    f"renorm_wrong_prob: {wrong_prob:.8f}",
                    f"correct: {is_correct}",
                    "",
                ]
            )
    return "\n".join(lines)


def build_50_sample_logs(
    intervenable: IntervenableModel,
    sample_examples: list[dict[str, Any]],
    *,
    pad_id: int,
    intervene_direction: str,
    device: torch.device,
    header: str,
) -> str:
    return build_sample_logs(
        intervenable,
        sample_examples,
        pad_id=pad_id,
        intervene_direction=intervene_direction,
        device=device,
        header=header,
        expected_count=50,
    )


def build_epoch_tracking_text(
    *,
    intervene_direction: str,
    baseline_test_metrics: dict[str, float],
    epoch_tracking_history: list[dict[str, Any]],
    sample_logs_text: str,
    sample_count: int,
) -> str:
    lines: list[str] = [
        "Epoch-wise test tracking",
        f"Direction: {intervene_direction}",
        f"Fixed sample_count: {sample_count}",
        "",
        (
            "Baseline test metrics (before training): "
            f"test_acc={baseline_test_metrics['accuracy']:.6f}, "
            f"correct={baseline_test_metrics['correct']}, "
            f"total={baseline_test_metrics['total']}, "
            f"ties={baseline_test_metrics['ties']}"
        ),
        "",
        "Per-epoch test accuracy:",
    ]
    if not epoch_tracking_history:
        lines.append("- (no epochs completed yet)")
    else:
        for item in epoch_tracking_history:
            lines.append(
                (
                    f"- epoch={item['epoch']}, "
                    f"test_acc={item['test_accuracy']:.6f}, "
                    f"correct={item['test_correct']}, "
                    f"total={item['test_total']}, "
                    f"ties={item['test_ties']}"
                )
            )
    lines.extend(["", sample_logs_text])
    return "\n".join(lines)


def train(
    intervenable: IntervenableModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    training_cfg: dict[str, Any],
    *,
    device: torch.device,
    test_dataloader: DataLoader | None = None,
    epoch_end_callback: Callable[[dict[str, Any]], None] | None = None,
    best_rotation_checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    epochs = int(training_cfg.get("epochs", 3))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    lr_rotate = float(training_cfg.get("lr_rotate", 1.0e-3))
    lr_boundary = float(training_cfg.get("lr_boundary", 1.0e-2))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    temperature_start = float(training_cfg.get("temperature_start", 50.0))
    temperature_end = float(training_cfg.get("temperature_end", 0.1))
    boundary_loss_weight = float(training_cfg.get("boundary_loss_weight", 1.0))
    log_every_steps = int(training_cfg.get("log_every_steps", 20))
    early_stopping_cfg = normalize_nested_config(training_cfg.get("early_stopping", {}))
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 3))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.005))
    monitor_metric_name = str(
        early_stopping_cfg.get("monitor_metric", "test_accuracy")
    ).strip()
    checkpoint_cfg = normalize_nested_config(
        training_cfg.get("best_rotation_checkpoint", {})
    )
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", False))

    if early_stopping_patience <= 0:
        raise ValueError("training.early_stopping.patience must be > 0.")
    if early_stopping_min_delta < 0:
        raise ValueError("training.early_stopping.min_delta must be >= 0.")
    if monitor_metric_name not in {"test_accuracy", "val_accuracy"}:
        raise ValueError(
            "training.early_stopping.monitor_metric must be 'test_accuracy' or 'val_accuracy'."
        )

    total_train_steps = max(1, len(train_dataloader) * epochs)
    warmup_steps = int(warmup_ratio * total_train_steps)

    optimizer_params = []
    for _, intervention in intervenable.interventions.items():
        optimizer_params.append(
            {"params": intervention.rotate_layer.parameters(), "lr": lr_rotate}
        )
        optimizer_params.append(
            {"params": intervention.intervention_boundaries, "lr": lr_boundary}
        )
    optimizer = torch.optim.Adam(optimizer_params, lr=lr_rotate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    temperature_schedule = torch.linspace(
        temperature_start,
        temperature_end,
        total_train_steps,
        device=device,
    )
    if device.type == "cuda":
        temperature_schedule = temperature_schedule.to(torch.bfloat16)

    intervenable.set_zero_grad()
    intervenable.set_temperature(temperature_schedule[0])
    intervenable.model.train()

    history: list[dict[str, Any]] = []
    global_step = 0
    track_best_state = early_stopping_enabled or checkpoint_enabled
    best_state: list[dict[str, Any]] | None = None
    best_monitor_accuracy: float | None = None
    best_epoch: int | None = None
    stale_eval_count = 0
    stopped_early = False
    stop_reason: str | None = None
    saved_checkpoint_path: str | None = None

    for epoch in trange(epochs, desc="Epoch"):
        epoch_total = 0
        epoch_correct = 0
        epoch_loss_sum = 0.0

        progress = tqdm(
            train_dataloader,
            desc=f"Train Epoch {epoch + 1}/{epochs}",
            leave=True,
        )
        for step, batch in enumerate(progress):
            schedule_idx = min(global_step, total_train_steps - 1)
            intervenable.set_temperature(temperature_schedule[schedule_idx])

            batch = move_batch_to_device(batch, device)
            pair_logits = forward_intervened_pair_logits(intervenable, batch)
            total_loss, ce_loss, boundary_penalty = compute_targeted_loss(
                intervenable,
                pair_logits,
                boundary_loss_weight=boundary_loss_weight,
            )

            batch_size = pair_logits.shape[0]
            margins = pair_logits[:, 0] - pair_logits[:, 1]
            batch_correct = (margins > 0).sum().item()
            epoch_total += batch_size
            epoch_correct += batch_correct
            epoch_loss_sum += total_loss.item() * batch_size

            loss_for_backward = total_loss / max(1, gradient_accumulation_steps)
            loss_for_backward.backward()

            should_step = (
                ((step + 1) % max(1, gradient_accumulation_steps) == 0)
                or (step + 1 == len(train_dataloader))
            )
            if should_step:
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()

            if (step == 0) or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                batch_acc = batch_correct / batch_size if batch_size else 0.0
                progress.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    ce=f"{ce_loss.item():.4f}",
                    boundary=f"{boundary_penalty.item():.4f}",
                    acc=f"{batch_acc:.4f}",
                )

            global_step += 1

        train_accuracy = epoch_correct / epoch_total if epoch_total else 0.0
        train_loss = epoch_loss_sum / epoch_total if epoch_total else 0.0
        val_metrics = evaluate(intervenable, val_dataloader, device)
        test_metrics = None
        if test_dataloader is not None:
            test_metrics = evaluate(intervenable, test_dataloader, device)
        if monitor_metric_name == "test_accuracy" and test_metrics is None:
            raise ValueError(
                "training.early_stopping.monitor_metric='test_accuracy' requires a test dataloader."
            )
        if monitor_metric_name == "test_accuracy":
            monitor_accuracy = float(test_metrics["accuracy"])
        else:
            monitor_accuracy = float(val_metrics["accuracy"])
        is_best_epoch = False
        if track_best_state:
            if (
                best_monitor_accuracy is None
                or monitor_accuracy >= best_monitor_accuracy + early_stopping_min_delta
            ):
                best_monitor_accuracy = monitor_accuracy
                best_epoch = epoch + 1
                stale_eval_count = 0
                is_best_epoch = True
                best_state = get_intervention_state(intervenable)
                if checkpoint_enabled and best_rotation_checkpoint_path is not None:
                    torch.save(
                        {
                            "epoch": best_epoch,
                            monitor_metric_name: best_monitor_accuracy,
                            "min_delta": early_stopping_min_delta,
                            "monitor_metric": monitor_metric_name,
                            "intervention_state": best_state,
                        },
                        best_rotation_checkpoint_path,
                    )
                    saved_checkpoint_path = str(best_rotation_checkpoint_path)
                    print(
                        f"[checkpoint] saved best rotation state to {best_rotation_checkpoint_path} "
                        f"(epoch={best_epoch}, {monitor_metric_name}={best_monitor_accuracy:.4f})"
                    )
            else:
                stale_eval_count += 1

        epoch_result = {
            "epoch": epoch + 1,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_targeted_ce_loss": val_metrics["avg_targeted_ce_loss"],
            "val_avg_target_minus_wrong_logit": val_metrics[
                "avg_target_minus_wrong_logit"
            ],
            "is_best": is_best_epoch,
            "best_monitor_metric": monitor_metric_name,
            "best_test_accuracy": best_monitor_accuracy
            if monitor_metric_name == "test_accuracy"
            else None,
            "best_val_accuracy": best_monitor_accuracy
            if monitor_metric_name == "val_accuracy"
            else None,
            "early_stopping_stale_evals": stale_eval_count,
        }
        if test_metrics is not None:
            epoch_result["test_accuracy"] = test_metrics["accuracy"]
            epoch_result["test_targeted_ce_loss"] = test_metrics["avg_targeted_ce_loss"]
            epoch_result["test_avg_target_minus_wrong_logit"] = test_metrics[
                "avg_target_minus_wrong_logit"
            ]
            epoch_result["test_correct"] = test_metrics["correct"]
            epoch_result["test_total"] = test_metrics["total"]
            epoch_result["test_ties"] = test_metrics["ties"]
        history.append(epoch_result)

        message = (
            f"[epoch {epoch + 1}] train_acc={train_accuracy:.4f} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_ce={val_metrics['avg_targeted_ce_loss']:.4f}"
        )
        if test_metrics is not None:
            message += (
                f" test_acc={test_metrics['accuracy']:.4f} "
                f"test_ce={test_metrics['avg_targeted_ce_loss']:.4f}"
            )
        print(message)
        if epoch_end_callback is not None:
            epoch_end_callback(epoch_result)

        if early_stopping_enabled and stale_eval_count >= early_stopping_patience:
            stopped_early = True
            stop_reason = (
                f"{monitor_metric_name} did not improve by at least "
                f"{early_stopping_min_delta} for {early_stopping_patience} evaluations"
            )
            print(f"[early-stop] {stop_reason}; stopping at epoch {epoch + 1}")
            break

    loaded_best_state = False
    if best_state is not None:
        load_intervention_state(intervenable, best_state)
        loaded_best_state = True
        print(
            f"[best] restored best intervention state from epoch {best_epoch} "
            f"({monitor_metric_name}={best_monitor_accuracy:.4f})"
        )

    training_summary = {
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "monitor_metric": monitor_metric_name if best_epoch is not None else None,
            "best_test_accuracy": best_monitor_accuracy
            if best_epoch is not None and monitor_metric_name == "test_accuracy"
            else None,
            "best_val_accuracy": best_monitor_accuracy
            if best_epoch is not None and monitor_metric_name == "val_accuracy"
            else None,
            "stale_evals": stale_eval_count,
        },
        "best_rotation_checkpoint": {
            "enabled": checkpoint_enabled,
            "path": saved_checkpoint_path,
            "loaded_best_state": loaded_best_state,
        },
    }
    return history, training_summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    runtime_cfg = config.get("runtime", {})
    intervention_cfg = config.get("intervention", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    model_path = str(model_cfg.get("path", "")).strip()
    if not model_path:
        raise ValueError("model.path is required in config.")
    model_name = str(model_cfg.get("name", model_path)).strip()

    intervene_direction = str(
        intervention_cfg.get("intervene_direction", "source_to_base")
    ).strip()
    if intervene_direction not in {"source_to_base", "base_to_source"}:
        raise ValueError(
            "intervene_direction must be 'source_to_base' or 'base_to_source'."
        )

    split_seed = int(dataset_cfg.get("shuffle_seed", 42))
    train_size_value = dataset_cfg.get("train_size", None)
    train_size = int(train_size_value) if train_size_value is not None else None
    val_size = int(dataset_cfg.get("val_size", 1000))
    test_size = int(dataset_cfg.get("test_size", 0))
    train_seed = int(training_cfg.get("seed", 42))
    random_rotation_eval = bool(
        training_cfg.get("random_rotation_eval", training_cfg.get("skip_training", False))
    )

    set_seed(train_seed)
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer, model, runtime_device, device_map = load_model_and_tokenizer(
        model_path,
        runtime_cfg,
        hf_token,
    )
    model_device = next(model.parameters()).device

    dataset_runs = build_dataset_run_specs(dataset_cfg, config_dir)
    dataset_name = str(dataset_cfg.get("dataset_name", "")).strip()
    variation = str(dataset_cfg.get("variation", "")).strip()
    print(
        f"[data] dataset={dataset_name} variation={variation} "
        f"nua_values={[run['nua'] for run in dataset_runs]}"
    )
    print(f"[setup] intervene_direction={intervene_direction}")
    if random_rotation_eval:
        print("[setup] random_rotation_eval=true; evaluating initialized rotation without training")

    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    raw_layer_value = intervention_cfg.get("layer", intervention_cfg.get("layers", 15))
    if isinstance(raw_layer_value, (list, tuple)):
        layers = [int(layer_item) for layer_item in raw_layer_value]
    else:
        layers = [int(raw_layer_value)]
    if not layers:
        raise ValueError("intervention.layer must include at least one layer.")
    component = str(intervention_cfg.get("component", "block_output"))
    unit = str(intervention_cfg.get("unit", "pos"))
    print(f"[setup] layers={layers}")

    output_root = resolve_path(config_dir, str(output_cfg.get("root_dir", "results")))
    checkpoint_cfg = normalize_nested_config(
        training_cfg.get("best_rotation_checkpoint", {})
    )
    checkpoint_root = resolve_path(
        config_dir,
        str(checkpoint_cfg.get("root_dir", "checkpoints")),
    )
    model_dir_name = sanitize_model_dir_name(model_name, model_path)
    epoch_tracking_cfg = output_cfg.get("epoch_tracking", {})
    epoch_tracking_enabled = bool(epoch_tracking_cfg.get("enabled", True))
    epoch_tracking_sample_count = int(epoch_tracking_cfg.get("sample_count", 10))
    epoch_tracking_file_name = str(
        epoch_tracking_cfg.get("file_name", "epoch_test_tracking.txt")
    )
    if epoch_tracking_sample_count <= 0:
        raise ValueError("output.epoch_tracking.sample_count must be > 0")

    variation_result_dir = output_root / model_dir_name / dataset_name / variation
    variation_checkpoint_dir = checkpoint_root / model_dir_name / dataset_name / variation
    variation_result_dir.mkdir(parents=True, exist_ok=True)

    all_nua_summary: list[dict[str, Any]] = []
    for run_spec in dataset_runs:
        nua = int(run_spec["nua"])
        train_path = Path(run_spec["train_path"])
        test_path = Path(run_spec["test_path"])
        attractor_dir = variation_result_dir / f"{nua}_attractor" / intervene_direction
        attractor_checkpoint_dir = (
            variation_checkpoint_dir / f"{nua}_attractor" / intervene_direction
        )
        attractor_dir.mkdir(parents=True, exist_ok=True)

        train_rows = load_jsonl(train_path)
        test_rows = load_jsonl(test_path)
        train_pool_examples = build_examples(train_rows, tokenizer)
        test_examples_full = build_examples(test_rows, tokenizer)

        if test_size > 0:
            if len(test_examples_full) < test_size:
                raise ValueError(
                    f"Test file {test_path} has only {len(test_examples_full)} rows, "
                    f"but dataset.test_size={test_size} was requested."
                )
            test_examples = test_examples_full[:test_size]
        else:
            test_examples = test_examples_full

        train_examples, val_examples = split_train_val_examples(
            train_pool_examples,
            val_size=val_size,
            seed=split_seed + nua,
            train_size=train_size,
        )
        if len(test_examples) < 50:
            raise ValueError(
                f"Need at least 50 test rows for sample logging, got {len(test_examples)} for {test_path}"
            )
        if epoch_tracking_sample_count > len(test_examples):
            raise ValueError(
                "output.epoch_tracking.sample_count cannot exceed test set size "
                f"({len(test_examples)})."
            )

        print(
            f"[data][nua={nua}] train_file={train_path.name} test_file={test_path.name} "
            f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}"
        )

        collate_fn = lambda batch: collate_examples(
            batch,
            pad_id=pad_id,
            intervene_direction=intervene_direction,
        )
        base_train_batch_size = int(training_cfg.get("batch_size", 16))
        base_eval_batch_size = int(training_cfg.get("eval_batch_size", 16))
        min_batch_size = int(training_cfg.get("min_batch_size", 1))
        current_train_batch_size = get_auto_halved_batch_size(
            base_train_batch_size,
            nua,
            min_batch_size=min_batch_size,
        )
        current_eval_batch_size = get_auto_halved_batch_size(
            base_eval_batch_size,
            nua,
            min_batch_size=min_batch_size,
        )
        print(
            f"[batch][nua={nua}] train_bs={current_train_batch_size} "
            f"eval_bs={current_eval_batch_size} "
            f"(base_train_bs={base_train_batch_size}, base_eval_bs={base_eval_batch_size})"
        )
        train_dataloader = DataLoader(
            train_examples,
            batch_size=current_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            val_examples,
            batch_size=current_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_dataloader = DataLoader(
            test_examples,
            batch_size=current_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        sample_rng = random.Random(split_seed + 999 + nua)
        sample_indices = sample_rng.sample(range(len(test_examples)), 50)
        sample_examples = [test_examples[i] for i in sample_indices]
        epoch_tracking_rng = random.Random(split_seed + 2026 + nua)
        epoch_tracking_indices = epoch_tracking_rng.sample(
            range(len(test_examples)), epoch_tracking_sample_count
        )
        epoch_tracking_examples = [test_examples[i] for i in epoch_tracking_indices]

        all_layer_results: list[dict[str, Any]] = []
        for layer in layers:
            set_seed(train_seed)
            random.seed(train_seed)
            torch.manual_seed(train_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(train_seed)

            layer_result_dir = attractor_dir / f"layer_{layer}"
            layer_result_dir.mkdir(parents=True, exist_ok=True)
            epoch_tracking_path = layer_result_dir / epoch_tracking_file_name
            layer_checkpoint_dir = attractor_checkpoint_dir / f"layer_{layer}"
            if bool(checkpoint_cfg.get("enabled", False)):
                layer_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_rotation_checkpoint_path = layer_checkpoint_dir / str(
                checkpoint_cfg.get("file_name", "best_rotation_checkpoint.pt")
            )

            intervenable_config = IntervenableConfig(
                model_type=type(model),
                representations=[
                    RepresentationConfig(
                        layer=layer,
                        component=component,
                        unit=unit,
                        max_number_of_units=1,
                    )
                ],
                intervention_types=BoundlessRotatedSpaceIntervention,
            )
            intervenable = IntervenableModel(intervenable_config, model)
            intervenable.disable_model_gradients()

            if device_map is None:
                intervenable.set_device(runtime_device)
                training_device = runtime_device
            else:
                intervenable.set_device(model_device, set_model=False)
                training_device = model_device

            print(
                f"[model][nua={nua}][layer {layer}] base model params: {count_parameters(intervenable.model)}"
            )
            print(
                f"[model][nua={nua}][layer {layer}] intervention params: {intervenable.count_parameters()}"
            )

            before_header = (
                "Detailed 50-sample logs BEFORE training\n"
                f"Direction: {intervene_direction}\n"
                f"NUA: {nua}\n"
                f"Layer: {layer}\n"
                "Logit index 0 = target correct verb, index 1 = wrong verb"
            )
            before_text = build_50_sample_logs(
                intervenable,
                sample_examples,
                pad_id=pad_id,
                intervene_direction=intervene_direction,
                device=training_device,
                header=before_header,
            )
            before_path = layer_result_dir / str(
                output_cfg.get("before_training_log", "50_samples_before_training.txt")
            )
            before_path.write_text(before_text, encoding="utf-8")
            # print(before_text)

            baseline_val_metrics = evaluate(intervenable, val_dataloader, training_device)
            baseline_test_metrics = evaluate(intervenable, test_dataloader, training_device)
            print(
                f"[baseline][nua={nua}][layer {layer}] val_acc={baseline_val_metrics['accuracy']:.4f} "
                f"test_acc={baseline_test_metrics['accuracy']:.4f}"
            )

            epoch_tracking_history: list[dict[str, Any]] = []
            if epoch_tracking_enabled:
                baseline_tracking_header = (
                    "Fixed sample logs for epoch-wise test tracking (BEFORE training)\n"
                    f"Direction: {intervene_direction}\n"
                    f"NUA: {nua}\n"
                    f"Layer: {layer}\n"
                    "Logit index 0 = target correct verb, index 1 = wrong verb"
                )
                baseline_tracking_samples = build_sample_logs(
                    intervenable,
                    epoch_tracking_examples,
                    pad_id=pad_id,
                    intervene_direction=intervene_direction,
                    device=training_device,
                    header=baseline_tracking_header,
                    expected_count=epoch_tracking_sample_count,
                )
                tracking_text = build_epoch_tracking_text(
                    intervene_direction=intervene_direction,
                    baseline_test_metrics=baseline_test_metrics,
                    epoch_tracking_history=epoch_tracking_history,
                    sample_logs_text=baseline_tracking_samples,
                    sample_count=epoch_tracking_sample_count,
                )
                epoch_tracking_path.write_text(tracking_text, encoding="utf-8")
                print(f"[saved] {epoch_tracking_path} (before training)")

            def on_epoch_end(epoch_result: dict[str, Any]) -> None:
                if not epoch_tracking_enabled:
                    return
                epoch_tracking_history.append(
                    {
                        "epoch": int(epoch_result["epoch"]),
                        "test_accuracy": float(epoch_result["test_accuracy"]),
                        "test_correct": int(epoch_result["test_correct"]),
                        "test_total": int(epoch_result["test_total"]),
                        "test_ties": int(epoch_result["test_ties"]),
                    }
                )
                epoch_header = (
                    "Fixed sample logs for epoch-wise test tracking "
                    f"(AFTER epoch {epoch_result['epoch']})\n"
                    f"Direction: {intervene_direction}\n"
                    f"NUA: {nua}\n"
                    f"Layer: {layer}\n"
                    "Logit index 0 = target correct verb, index 1 = wrong verb"
                )
                epoch_samples = build_sample_logs(
                    intervenable,
                    epoch_tracking_examples,
                    pad_id=pad_id,
                    intervene_direction=intervene_direction,
                    device=training_device,
                    header=epoch_header,
                    expected_count=epoch_tracking_sample_count,
                )
                tracking_text = build_epoch_tracking_text(
                    intervene_direction=intervene_direction,
                    baseline_test_metrics=baseline_test_metrics,
                    epoch_tracking_history=epoch_tracking_history,
                    sample_logs_text=epoch_samples,
                    sample_count=epoch_tracking_sample_count,
                )
                epoch_tracking_path.write_text(tracking_text, encoding="utf-8")
                print(
                    f"[saved] {epoch_tracking_path} (after epoch {epoch_result['epoch']})"
                )

            if random_rotation_eval:
                history = []
                training_summary = {
                    "early_stopping": {
                        "enabled": False,
                        "patience": int(
                            normalize_nested_config(
                                training_cfg.get("early_stopping", {})
                            ).get("patience", 3)
                        ),
                        "min_delta": float(
                            normalize_nested_config(
                                training_cfg.get("early_stopping", {})
                            ).get("min_delta", 0.005)
                        ),
                        "stopped_early": False,
                        "stop_reason": "random_rotation_eval skips training",
                        "best_epoch": None,
                        "monitor_metric": None,
                        "best_test_accuracy": None,
                        "best_val_accuracy": None,
                        "stale_evals": 0,
                    },
                    "best_rotation_checkpoint": {
                        "enabled": bool(checkpoint_cfg.get("enabled", False)),
                        "path": None,
                        "loaded_best_state": False,
                    },
                }
                final_val_metrics = baseline_val_metrics
                final_test_metrics = baseline_test_metrics
                print(
                    f"[random][nua={nua}][layer {layer}] val_acc={final_val_metrics['accuracy']:.4f} "
                    f"test_acc={final_test_metrics['accuracy']:.4f}"
                )
                after_header = (
                    "Detailed 50-sample logs RANDOM ROTATION (no training)\n"
                    f"Direction: {intervene_direction}\n"
                    f"NUA: {nua}\n"
                    f"Layer: {layer}\n"
                    "Logit index 0 = target correct verb, index 1 = wrong verb"
                )
            else:
                history, training_summary = train(
                    intervenable,
                    train_dataloader,
                    val_dataloader,
                    training_cfg,
                    device=training_device,
                    test_dataloader=test_dataloader,
                    epoch_end_callback=on_epoch_end,
                    best_rotation_checkpoint_path=best_rotation_checkpoint_path,
                )

                final_val_metrics = evaluate(intervenable, val_dataloader, training_device)
                final_test_metrics = evaluate(intervenable, test_dataloader, training_device)
                print(
                    f"[final][nua={nua}][layer {layer}] val_acc={final_val_metrics['accuracy']:.4f} "
                    f"test_acc={final_test_metrics['accuracy']:.4f}"
                )

                after_header = (
                    "Detailed 50-sample logs AFTER training\n"
                    f"Direction: {intervene_direction}\n"
                    f"NUA: {nua}\n"
                    f"Layer: {layer}\n"
                    "Logit index 0 = target correct verb, index 1 = wrong verb"
                )
            after_text = build_50_sample_logs(
                intervenable,
                sample_examples,
                pad_id=pad_id,
                intervene_direction=intervene_direction,
                device=training_device,
                header=after_header,
            )
            after_path = layer_result_dir / str(
                output_cfg.get("after_training_log", "50_samples_after_training.txt")
            )
            after_path.write_text(after_text, encoding="utf-8")
            # print(after_text)

            result_payload = {
                "model": {
                    "name": model_name,
                    "path": model_path,
                    "output_dir_name": model_dir_name,
                },
                "dataset": {
                    "dataset_name": dataset_name,
                    "variation": variation,
                    "nua": nua,
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                    "train_pool_size": len(train_pool_examples),
                    "train_size": len(train_examples),
                    "val_size": len(val_examples),
                    "test_size": len(test_examples),
                    "shuffle_seed": split_seed,
                    "sample_row_indices_for_50_logs": [
                        sample_examples[i]["row_idx"] for i in range(50)
                    ],
                    "sample_row_indices_for_epoch_tracking_logs": [
                        epoch_tracking_examples[i]["row_idx"]
                        for i in range(epoch_tracking_sample_count)
                    ],
                },
                "intervention": {
                    "direction": intervene_direction,
                    "layer": layer,
                    "component": component,
                    "unit": unit,
                    "position": "last_token_of_prefix",
                },
                "training": {
                    "seed": train_seed,
                    "random_rotation_eval": random_rotation_eval,
                    "trained": not random_rotation_eval,
                    "epochs_completed": len(history),
                    "epochs": int(training_cfg.get("epochs", 3)),
                    "batch_size": current_train_batch_size,
                    "eval_batch_size": current_eval_batch_size,
                    "base_batch_size": base_train_batch_size,
                    "base_eval_batch_size": base_eval_batch_size,
                    "batch_size_schedule": "half_per_nua_increment",
                    "gradient_accumulation_steps": int(
                        training_cfg.get("gradient_accumulation_steps", 1)
                    ),
                    "lr_rotate": float(training_cfg.get("lr_rotate", 1.0e-3)),
                    "lr_boundary": float(training_cfg.get("lr_boundary", 1.0e-2)),
                    "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
                    "temperature_start": float(
                        training_cfg.get("temperature_start", 50.0)
                    ),
                    "temperature_end": float(training_cfg.get("temperature_end", 0.1)),
                    "boundary_loss_weight": float(
                        training_cfg.get("boundary_loss_weight", 1.0)
                    ),
                    "early_stopping": training_summary["early_stopping"],
                    "best_rotation_checkpoint": training_summary[
                        "best_rotation_checkpoint"
                    ],
                    "history": history,
                },
                "baseline": {
                    "val": baseline_val_metrics,
                    "test": baseline_test_metrics,
                },
                "final": {
                    "val": final_val_metrics,
                    "test": final_test_metrics,
                },
                "artifacts": {
                    "before_training_log": str(before_path),
                    "after_training_log": str(after_path),
                    "epoch_tracking_log": (
                        str(epoch_tracking_path) if epoch_tracking_enabled else None
                    ),
                    "best_rotation_checkpoint": training_summary[
                        "best_rotation_checkpoint"
                    ]["path"],
                },
            }
            result_path = layer_result_dir / str(
                output_cfg.get("result_file", "result.json")
            )
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(result_payload, handle, indent=2)
            print(f"[saved] {result_path}")

            all_layer_results.append(
                {
                    "layer": layer,
                    "result_file": str(result_path),
                    "final_test_accuracy": final_test_metrics["accuracy"],
                    "final_val_accuracy": final_val_metrics["accuracy"],
                }
            )

        summary_path = attractor_dir / "all_layers_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump({"layers": all_layer_results, "nua": nua}, handle, indent=2)
        print(f"[saved] {summary_path}")
        all_nua_summary.append(
            {
                "nua": nua,
                "summary_file": str(summary_path),
                "result_root": str(attractor_dir),
            }
        )

    overall_summary_path = (
        variation_result_dir / f"{intervene_direction}_all_nua_summary.json"
    )
    with overall_summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": all_nua_summary}, handle, indent=2)
    print(f"[saved] {overall_summary_path}")


if __name__ == "__main__":
    main()