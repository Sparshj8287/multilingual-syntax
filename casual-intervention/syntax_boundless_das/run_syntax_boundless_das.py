import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PYVENE_ROOT = os.path.join(REPO_ROOT, "pyvene")
if PYVENE_ROOT not in sys.path:
    sys.path.insert(0, PYVENE_ROOT)

from pyvene import (  # noqa: E402
    BoundlessRotatedSpaceIntervention,
    IntervenableConfig,
    IntervenableModel,
    RepresentationConfig,
    set_seed,
)


def load_yaml_config(path: str | None) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def resolve_device_map(device_map: str | None):
    if not device_map:
        return None
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


def normalize_models(models_cfg: List | None) -> List[Dict[str, str]]:
    if not models_cfg:
        return []
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


def load_model(
    model_path: str,
    hf_token: str | None,
    device_map: str | None,
    torch_dtype,
    device: torch.device,
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
        device_map=device_map,
        token=hf_token,
    )
    if model.get_input_embeddings().num_embeddings < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if device_map is None:
        model.to(device)
    model.eval()
    return tokenizer, model


def encode_sentence(tokenizer: AutoTokenizer, sentence: str) -> List[int]:
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        if not ids or ids[0] != tokenizer.bos_token_id:
            ids = [tokenizer.bos_token_id] + ids
    return ids


def build_padded_batch(
    token_ids_list: List[List[int]],
    pad_id: int,
    max_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    max_len = max_len or max(len(ids) for ids in token_ids_list)
    input_ids = []
    attention_mask = []
    seq_lens = []
    for ids in token_ids_list:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        seq_lens.append(len(ids))
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        seq_lens,
    )


def sequence_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    shifted_log_probs = log_probs[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    shifted_mask = attention_mask[:, 1:].float()
    token_logprobs = torch.gather(
        shifted_log_probs, dim=-1, index=shifted_targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logprobs * shifted_mask).sum(dim=-1)


def normalize_token_positions(value) -> List[str]:
    if isinstance(value, str):
        if value == "both":
            return ["start", "end"]
        return [value]
    if not value:
        return ["start"]
    if "both" in value:
        return ["start", "end"]
    return value


def build_positions(
    lengths: List[int],
    token_positions: List[str],
    token_index: int,
) -> List[List[int]]:
    positions = []
    num_positions = len(token_positions)
    for seq_len in lengths:
        if seq_len <= 0:
            positions.append([0] * num_positions)
            continue
        pos_list = []
        for pos_type in token_positions:
            if pos_type == "start":
                if token_index < 0:
                    start_pos = max(seq_len + token_index, 0)
                else:
                    start_pos = min(token_index, seq_len - 1)
                pos_list.append(start_pos)
            elif pos_type == "end":
                pos_list.append(seq_len - 1)
            else:
                raise ValueError(f"Unsupported token position: {pos_type}")
        positions.append(pos_list)
    return positions


def build_unit_locations(
    source_positions: List[List[int]],
    base_positions: List[List[int]],
) -> Dict:
    return {"sources->base": ([source_positions], [base_positions])}


def collate_batch(
    examples: List[Dict],
    tokenizer: AutoTokenizer,
    pad_id: int,
) -> Dict[str, torch.Tensor | List[int]]:
    good_ids = [encode_sentence(tokenizer, ex["sentence_good"]) for ex in examples]
    bad_ids = [encode_sentence(tokenizer, ex["sentence_bad"]) for ex in examples]

    good_input_ids, good_attention_mask, good_lens = build_padded_batch(
        good_ids, pad_id, max_len=max(
            max(len(ids) for ids in good_ids),
            max(len(ids) for ids in bad_ids),
        )
    )
    bad_input_ids, bad_attention_mask, bad_lens = build_padded_batch(
        bad_ids, pad_id, max_len=good_input_ids.shape[1]
    )

    return {
        "good_input_ids": good_input_ids,
        "good_attention_mask": good_attention_mask,
        "good_lengths": good_lens,
        "bad_input_ids": bad_input_ids,
        "bad_attention_mask": bad_attention_mask,
        "bad_lengths": bad_lens,
    }


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def load_blimp_subset(
    dataset_name: str,
    uids: List[str],
    shuffle_seed: int,
    val_size: int,
    test_size: int,
):
    datasets = [load_dataset(dataset_name, uid, split="train") for uid in uids]
    dataset = concatenate_datasets(datasets).shuffle(seed=shuffle_seed)
    if len(dataset) < val_size + test_size:
        raise ValueError(
            "Dataset too small for requested splits: "
            f"{len(dataset)} < {val_size + test_size}"
        )
    test_split = dataset.select(range(0, test_size))
    val_split = dataset.select(range(test_size, test_size + val_size))
    train_split = dataset.select(range(test_size + val_size, len(dataset)))
    return train_split, val_split, test_split


def compute_pairwise_loss(
    logp_good: torch.Tensor, logp_bad: torch.Tensor
) -> torch.Tensor:
    diff = logp_good - logp_bad
    return -torch.nn.functional.logsigmoid(diff).mean()


def forward_intervened_logprobs(
    intervenable: IntervenableModel,
    batch: Dict,
    token_positions: List[str],
    token_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    good_input_ids = batch["good_input_ids"]
    good_attention_mask = batch["good_attention_mask"]
    bad_input_ids = batch["bad_input_ids"]
    bad_attention_mask = batch["bad_attention_mask"]
    good_lengths = batch["good_lengths"]
    bad_lengths = batch["bad_lengths"]

    good_positions = build_positions(good_lengths, token_positions, token_index)
    bad_positions = build_positions(bad_lengths, token_positions, token_index)

    base_input_ids = torch.cat([good_input_ids, bad_input_ids], dim=0)
    base_attention_mask = torch.cat([good_attention_mask, bad_attention_mask], dim=0)
    source_input_ids = torch.cat([bad_input_ids, good_input_ids], dim=0)
    source_attention_mask = torch.cat([bad_attention_mask, good_attention_mask], dim=0)

    base_positions = good_positions + bad_positions
    source_positions = bad_positions + good_positions

    unit_locations = build_unit_locations(source_positions, base_positions)

    _, counterfactual_outputs = intervenable(
        {"input_ids": base_input_ids, "attention_mask": base_attention_mask},
        [{"input_ids": source_input_ids, "attention_mask": source_attention_mask}],
        unit_locations,
    )

    logits = counterfactual_outputs.logits
    batch_size = good_input_ids.shape[0]
    logits_good = logits[:batch_size]
    logits_bad = logits[batch_size:]

    logp_good = sequence_logprobs(logits_good, good_input_ids, good_attention_mask)
    logp_bad = sequence_logprobs(logits_bad, bad_input_ids, bad_attention_mask)
    return logp_good, logp_bad


def evaluate(
    intervenable: IntervenableModel,
    dataloader: DataLoader,
    token_positions: List[str],
    token_index: int,
    device: torch.device,
) -> Dict[str, float]:
    total_pairs = 0
    correct_pairs = 0
    sum_logprob_diff = 0.0
    sum_loss = 0.0

    intervenable.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logp_good, logp_bad = forward_intervened_logprobs(
                intervenable, batch, token_positions, token_index
            )
            diff = logp_good - logp_bad
            loss_value = compute_pairwise_loss(logp_good, logp_bad).item()
            total_pairs += diff.numel()
            correct_pairs += (diff > 0).sum().item()
            sum_logprob_diff += diff.sum().item()
            sum_loss += loss_value * diff.numel()

    wrong_pairs = total_pairs - correct_pairs
    accuracy = correct_pairs / total_pairs if total_pairs else 0.0
    avg_logprob_diff = sum_logprob_diff / total_pairs if total_pairs else 0.0
    avg_loss = sum_loss / total_pairs if total_pairs else 0.0
    return {
        "total_pairs": total_pairs,
        "correct_pairs": correct_pairs,
        "wrong_pairs": wrong_pairs,
        "accuracy": accuracy,
        "avg_logprob_diff": avg_logprob_diff,
        "avg_loss": avg_loss,
    }


def train_layer(
    intervenable: IntervenableModel,
    train_dataloader: DataLoader,
    token_positions: List[str],
    token_index: int,
    device: torch.device,
    training_cfg: Dict,
    eval_dataloader: Optional[DataLoader] = None,
) -> None:
    epochs = int(training_cfg.get("epochs", 3))
    gradient_accumulation_steps = int(
        training_cfg.get("gradient_accumulation_steps", 4)
    )
    log_every = int(training_cfg.get("log_every", 50))
    lr_rotate = float(training_cfg.get("lr_rotate", 1.0e-3))
    lr_boundary = float(training_cfg.get("lr_boundary", 1.0e-2))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    temperature_start = float(training_cfg.get("temperature_start", 50.0))
    temperature_end = float(training_cfg.get("temperature_end", 0.1))
    boundary_loss_weight = float(training_cfg.get("boundary_loss_weight", 1.0))

    t_total = int(len(train_dataloader) * epochs)
    warm_up_steps = int(warmup_ratio * t_total)
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, t_total)
        .to(torch.bfloat16)
        .to(device)
    )
    intervenable.set_temperature(temperature_schedule[0])

    optimizer_params = []
    for _, v in intervenable.interventions.items():
        optimizer_params.append({"params": v.rotate_layer.parameters(), "lr": lr_rotate})
        optimizer_params.append({"params": v.intervention_boundaries, "lr": lr_boundary})
    optimizer = torch.optim.Adam(optimizer_params, lr=lr_rotate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    total_step = 0
    intervenable.model.train()
    for epoch in range(epochs):
        print(f"[train] epoch {epoch + 1}/{epochs} start")
        epoch_loss_sum = 0.0
        epoch_examples = 0
        progress = tqdm(
            train_dataloader,
            desc=f"train epoch {epoch + 1}/{epochs}",
            leave=True,
        )
        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)
            logp_good, logp_bad = forward_intervened_logprobs(
                intervenable, batch, token_positions, token_index
            )
            loss = compute_pairwise_loss(logp_good, logp_bad)
            boundary_loss = 0.0
            for _, v in intervenable.interventions.items():
                boundary_loss = boundary_loss + v.intervention_boundaries.sum()
            loss = loss + boundary_loss_weight * boundary_loss

            raw_loss_value = loss.item()
            epoch_loss_sum += raw_loss_value * logp_good.numel()
            epoch_examples += logp_good.numel()

            if step == 0 or (log_every > 0 and step % log_every == 0):
                progress.set_postfix(loss=f"{raw_loss_value:.4f}")

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    intervenable.set_temperature(temperature_schedule[total_step])
            total_step += 1
        avg_epoch_loss = (
            epoch_loss_sum / epoch_examples if epoch_examples else 0.0
        )
        print(f"[train] epoch {epoch + 1}/{epochs} avg_loss={avg_epoch_loss:.4f}")
        if eval_dataloader is not None:
            eval_metrics = evaluate(
                intervenable,
                eval_dataloader,
                token_positions,
                token_index,
                device,
            )
            print(
                f"[eval] epoch {epoch + 1}/{epochs} "
                f"acc={eval_metrics['accuracy']:.4f} "
                f"avg_loss={eval_metrics['avg_loss']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Boundless DAS for BLiMP syntax preference."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(SCRIPT_DIR, "config.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    intervention_cfg = config.get("intervention", {})
    runtime_cfg = config.get("runtime", {})
    output_cfg = config.get("output", {})

    dataset_name = dataset_cfg.get("name", "nyu-mll/blimp")
    uids = dataset_cfg.get("uids", [])
    val_size = int(dataset_cfg.get("val_size", 1000))
    test_size = int(dataset_cfg.get("test_size", 1000))
    shuffle_seed = int(dataset_cfg.get("shuffle_seed", 42))

    if not uids:
        raise ValueError("No BLiMP UIDs provided in config.")

    models = normalize_models(config.get("models"))
    if not models:
        raise ValueError("No models provided in config.")

    device = torch.device(runtime_cfg.get("device", "cuda"))
    device_map = resolve_device_map(runtime_cfg.get("device_map"))
    torch_dtype = resolve_torch_dtype(runtime_cfg.get("torch_dtype"))

    component = intervention_cfg.get("component", "block_output")
    unit = intervention_cfg.get("unit", "pos")
    layers = intervention_cfg.get("layers", [15])
    token_positions = normalize_token_positions(intervention_cfg.get("token_positions"))
    token_index = int(intervention_cfg.get("token_index", 0))

    train_data, val_data, test_data = load_blimp_subset(
        dataset_name=dataset_name,
        uids=uids,
        shuffle_seed=shuffle_seed,
        val_size=val_size,
        test_size=test_size,
    )
    print(
        f"[data] train={len(train_data)} val={len(val_data)} "
        f"test={len(test_data)}"
    )

    results_base_dir = os.path.join(
        SCRIPT_DIR, output_cfg.get("results_dir", "results")
    )
    os.makedirs(results_base_dir, exist_ok=True)

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for model_cfg in models:
        model_name = model_cfg["name"]
        model_path = model_cfg["path"]
        print(f"[model] loading {model_name} from {model_path}")
        tokenizer, model = load_model(
            model_path=model_path,
            hf_token=hf_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            device=device,
        )
        print(f"[model] loaded {model_name}")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        train_dataloader = DataLoader(
            train_data,
            batch_size=int(training_cfg.get("batch_size", 16)),
            shuffle=True,
            collate_fn=lambda ex: collate_batch(ex, tokenizer, pad_id),
        )
        eval_dataloader = DataLoader(
            val_data,
            batch_size=int(training_cfg.get("eval_batch_size", 16)),
            shuffle=False,
            collate_fn=lambda ex: collate_batch(ex, tokenizer, pad_id),
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=int(training_cfg.get("eval_batch_size", 16)),
            shuffle=False,
            collate_fn=lambda ex: collate_batch(ex, tokenizer, pad_id),
        )
        print(
            f"[data] batches train={len(train_dataloader)} "
            f"val={len(eval_dataloader)} test={len(test_dataloader)}"
        )

        model_dir = os.path.join(results_base_dir, sanitize_model_id(model_name))
        os.makedirs(model_dir, exist_ok=True)
        layer_results = []

        for layer in layers:
            set_seed(int(training_cfg.get("seed", 42)))
            print(
                f"[setup] model={model_name} layer={layer} "
                f"positions={token_positions} token_index={token_index}"
            )
            intervenable_config = IntervenableConfig(
                model_type=type(model),
                representations=[
                    RepresentationConfig(
                        layer=layer,
                        component=component,
                        unit=unit,
                        max_number_of_units=len(token_positions),
                    )
                ],
                intervention_types=BoundlessRotatedSpaceIntervention,
            )
            intervenable = IntervenableModel(intervenable_config, model)
            intervenable.disable_model_gradients()

            if device_map is None:
                intervenable.set_device(device)
            else:
                primary_device = next(model.parameters()).device
                intervenable.set_device(primary_device, set_model=False)

            for key, intervention in intervenable.interventions.items():
                if hasattr(intervention, "rotate_layer"):
                    weight_shape = tuple(intervention.rotate_layer.weight.shape)
                    print(
                        f"[info] {model_name} layer {layer} {key} rotate weight: {weight_shape}"
                    )

            print("[train] starting training")
            train_layer(
                intervenable,
                train_dataloader,
                token_positions,
                token_index,
                intervenable.get_device(),
                training_cfg,
                eval_dataloader=eval_dataloader,
            )
            print("[train] training complete, starting eval")

            val_metrics = evaluate(
                intervenable,
                eval_dataloader,
                token_positions,
                token_index,
                intervenable.get_device(),
            )
            test_metrics = evaluate(
                intervenable,
                test_dataloader,
                token_positions,
                token_index,
                intervenable.get_device(),
            )
            print(
                f"[eval] layer {layer} val_acc={val_metrics['accuracy']:.4f} "
                f"test_acc={test_metrics['accuracy']:.4f}"
            )

            layer_results.append(
                {
                    "layer": layer,
                    "val": val_metrics,
                    "test": test_metrics,
                }
            )

        result_payload = {
            "model": model_name,
            "model_path": model_path,
            "uids": uids,
            "layers": layer_results,
            "token_positions": token_positions,
            "token_index": token_index,
            "component": component,
            "unit": unit,
            "training": {
                "seed": int(training_cfg.get("seed", 42)),
                "epochs": int(training_cfg.get("epochs", 3)),
                "batch_size": int(training_cfg.get("batch_size", 16)),
                "eval_batch_size": int(training_cfg.get("eval_batch_size", 16)),
                "gradient_accumulation_steps": int(
                    training_cfg.get("gradient_accumulation_steps", 4)
                ),
                "lr_rotate": float(training_cfg.get("lr_rotate", 1.0e-3)),
                "lr_boundary": float(training_cfg.get("lr_boundary", 1.0e-2)),
                "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
                "temperature_start": float(training_cfg.get("temperature_start", 50.0)),
                "temperature_end": float(training_cfg.get("temperature_end", 0.1)),
                "boundary_loss_weight": float(
                    training_cfg.get("boundary_loss_weight", 1.0)
                ),
            },
            "dataset": {
                "name": dataset_name,
                "val_size": val_size,
                "test_size": test_size,
                "train_size": len(train_data),
            },
        }

        result_path = os.path.join(model_dir, "results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)


if __name__ == "__main__":
    main()
