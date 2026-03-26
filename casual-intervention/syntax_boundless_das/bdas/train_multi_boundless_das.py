#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import train_boundless_das as base

from pyvene import (  # noqa: E402
    IntervenableConfig,
    IntervenableModel,
    RepresentationConfig,
    count_parameters,
    set_seed,
)
from pyvene.models.interventions import (  # noqa: E402
    DistributedRepresentationIntervention,
    TrainableIntervention,
)
from pyvene.models.layers import RotateLayer  # noqa: E402


class FixedMaskRotatedSpaceIntervention(
    TrainableIntervention, DistributedRepresentationIntervention
):
    """
    Boundless-style rotated intervention with a fixed hard mask.

    Instead of learning a continuous boundary, this intervention always uses
    exactly `num_dim` dimensions in the rotated space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

        num_dim = int(kwargs.get("low_rank_dimension", 1))
        embed_dim = int(self.embed_dim)
        if num_dim <= 0 or num_dim > embed_dim:
            raise ValueError(
                f"`num_dim` must be in [1, {embed_dim}], got {num_dim}."
            )
        self.register_buffer("num_dim", torch.tensor(num_dim))

        fixed_mask = torch.zeros(embed_dim, dtype=torch.float32)
        fixed_mask[:num_dim] = 1.0
        self.register_buffer("fixed_mask", fixed_mask)

    def forward(self, base, source, subspaces=None, **kwargs):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Hard projection mask in rotated space: use first `num_dim` dimensions.
        mask = self.fixed_mask.to(rotated_base.dtype)
        mask_view = mask.view(*([1] * (rotated_base.dim() - 1)), -1)
        masked_diff = (rotated_source - rotated_base) * mask_view

        output = base + torch.matmul(masked_diff, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"FixedMaskRotatedSpaceIntervention(num_dim={int(self.num_dim)})"


def compute_targeted_loss(pair_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros(pair_logits.shape[0], dtype=torch.long, device=pair_logits.device)
    ce_loss = CrossEntropyLoss()(pair_logits, labels)
    return ce_loss, ce_loss


def train(
    intervenable: IntervenableModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    training_cfg: dict[str, Any],
    *,
    device: torch.device,
    test_dataloader: DataLoader | None = None,
    epoch_end_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    epochs = int(training_cfg.get("epochs", 3))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    lr_rotate = float(training_cfg.get("lr_rotate", 1.0e-3))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    log_every_steps = int(training_cfg.get("log_every_steps", 20))

    total_train_steps = max(1, len(train_dataloader) * epochs)
    warmup_steps = int(warmup_ratio * total_train_steps)

    optimizer_params = []
    for _, intervention in intervenable.interventions.items():
        optimizer_params.append(
            {"params": intervention.rotate_layer.parameters(), "lr": lr_rotate}
        )
    optimizer = torch.optim.Adam(optimizer_params, lr=lr_rotate)
    scheduler = base.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    intervenable.set_zero_grad()
    intervenable.model.train()

    history: list[dict[str, Any]] = []
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
            batch = base.move_batch_to_device(batch, device)
            pair_logits = base.forward_intervened_pair_logits(intervenable, batch)
            total_loss, ce_loss = compute_targeted_loss(pair_logits)

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
                    acc=f"{batch_acc:.4f}",
                )

        train_accuracy = epoch_correct / epoch_total if epoch_total else 0.0
        train_loss = epoch_loss_sum / epoch_total if epoch_total else 0.0
        val_metrics = base.evaluate(intervenable, val_dataloader, device)
        test_metrics = None
        if test_dataloader is not None:
            test_metrics = base.evaluate(intervenable, test_dataloader, device)

        epoch_result = {
            "epoch": epoch + 1,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_targeted_ce_loss": val_metrics["avg_targeted_ce_loss"],
            "val_avg_target_minus_wrong_logit": val_metrics[
                "avg_target_minus_wrong_logit"
            ],
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

    return history


def main() -> None:
    args = base.parse_args()
    config_path = Path(args.config).resolve()
    config = base.load_yaml(config_path)
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

    num_dim = int(intervention_cfg.get("num_dim", 1))
    if num_dim <= 0:
        raise ValueError("intervention.num_dim must be a positive integer.")

    split_seed = int(dataset_cfg.get("shuffle_seed", 42))
    train_size_value = dataset_cfg.get("train_size", None)
    train_size = int(train_size_value) if train_size_value is not None else None
    val_size = int(dataset_cfg.get("val_size", 1000))
    test_size = int(dataset_cfg.get("test_size", 0))
    train_seed = int(training_cfg.get("seed", 42))

    set_seed(train_seed)
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer, model, runtime_device, device_map = base.load_model_and_tokenizer(
        model_path,
        runtime_cfg,
        hf_token,
    )
    model_device = next(model.parameters()).device

    dataset_runs = base.build_dataset_run_specs(dataset_cfg, config_dir)
    dataset_name = str(dataset_cfg.get("dataset_name", "")).strip()
    variation = str(dataset_cfg.get("variation", "")).strip()
    print(
        f"[data] dataset={dataset_name} variation={variation} "
        f"nua_values={[run['nua'] for run in dataset_runs]}"
    )
    print(f"[setup] intervene_direction={intervene_direction}, num_dim={num_dim}")

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

    output_root = base.resolve_path(config_dir, str(output_cfg.get("root_dir", "results")))
    model_dir_name = base.sanitize_model_dir_name(model_name, model_path)
    epoch_tracking_cfg = output_cfg.get("epoch_tracking", {})
    epoch_tracking_enabled = bool(epoch_tracking_cfg.get("enabled", True))
    epoch_tracking_sample_count = int(epoch_tracking_cfg.get("sample_count", 10))
    epoch_tracking_file_name = str(
        epoch_tracking_cfg.get("file_name", "epoch_test_tracking.txt")
    )
    if epoch_tracking_sample_count <= 0:
        raise ValueError("output.epoch_tracking.sample_count must be > 0")

    variation_result_dir = output_root / model_dir_name / dataset_name / variation
    variation_result_dir.mkdir(parents=True, exist_ok=True)

    all_nua_summary: list[dict[str, Any]] = []
    for run_spec in dataset_runs:
        nua = int(run_spec["nua"])
        train_path = Path(run_spec["train_path"])
        test_path = Path(run_spec["test_path"])
        attractor_dir = variation_result_dir / f"{nua}_attractor" / intervene_direction
        attractor_dir.mkdir(parents=True, exist_ok=True)

        train_rows = base.load_jsonl(train_path)
        test_rows = base.load_jsonl(test_path)
        train_pool_examples = base.build_examples(train_rows, tokenizer)
        test_examples_full = base.build_examples(test_rows, tokenizer)

        if test_size > 0:
            if len(test_examples_full) < test_size:
                raise ValueError(
                    f"Test file {test_path} has only {len(test_examples_full)} rows, "
                    f"but dataset.test_size={test_size} was requested."
                )
            test_examples = test_examples_full[:test_size]
        else:
            test_examples = test_examples_full

        train_examples, val_examples = base.split_train_val_examples(
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

        collate_fn = lambda batch: base.collate_examples(
            batch,
            pad_id=pad_id,
            intervene_direction=intervene_direction,
        )
        base_train_batch_size = int(training_cfg.get("batch_size", 16))
        base_eval_batch_size = int(training_cfg.get("eval_batch_size", 16))
        min_batch_size = int(training_cfg.get("min_batch_size", 1))
        current_train_batch_size = base.get_auto_halved_batch_size(
            base_train_batch_size,
            nua,
            min_batch_size=min_batch_size,
        )
        current_eval_batch_size = base.get_auto_halved_batch_size(
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

            intervenable_config = IntervenableConfig(
                model_type=type(model),
                representations=[
                    RepresentationConfig(
                        layer=layer,
                        component=component,
                        unit=unit,
                        max_number_of_units=1,
                        low_rank_dimension=num_dim,
                    )
                ],
                intervention_types=FixedMaskRotatedSpaceIntervention,
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
                f"num_dim: {num_dim}\n"
                "Logit index 0 = target correct verb, index 1 = wrong verb"
            )
            before_text = base.build_50_sample_logs(
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
            print(before_text)

            baseline_val_metrics = base.evaluate(intervenable, val_dataloader, training_device)
            baseline_test_metrics = base.evaluate(intervenable, test_dataloader, training_device)
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
                    f"num_dim: {num_dim}\n"
                    "Logit index 0 = target correct verb, index 1 = wrong verb"
                )
                baseline_tracking_samples = base.build_sample_logs(
                    intervenable,
                    epoch_tracking_examples,
                    pad_id=pad_id,
                    intervene_direction=intervene_direction,
                    device=training_device,
                    header=baseline_tracking_header,
                    expected_count=epoch_tracking_sample_count,
                )
                tracking_text = base.build_epoch_tracking_text(
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
                    f"num_dim: {num_dim}\n"
                    "Logit index 0 = target correct verb, index 1 = wrong verb"
                )
                epoch_samples = base.build_sample_logs(
                    intervenable,
                    epoch_tracking_examples,
                    pad_id=pad_id,
                    intervene_direction=intervene_direction,
                    device=training_device,
                    header=epoch_header,
                    expected_count=epoch_tracking_sample_count,
                )
                tracking_text = base.build_epoch_tracking_text(
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

            history = train(
                intervenable,
                train_dataloader,
                val_dataloader,
                training_cfg,
                device=training_device,
                test_dataloader=test_dataloader,
                epoch_end_callback=on_epoch_end,
            )

            final_val_metrics = base.evaluate(intervenable, val_dataloader, training_device)
            final_test_metrics = base.evaluate(intervenable, test_dataloader, training_device)
            print(
                f"[final][nua={nua}][layer {layer}] val_acc={final_val_metrics['accuracy']:.4f} "
                f"test_acc={final_test_metrics['accuracy']:.4f}"
            )

            after_header = (
                "Detailed 50-sample logs AFTER training\n"
                f"Direction: {intervene_direction}\n"
                f"NUA: {nua}\n"
                f"Layer: {layer}\n"
                f"num_dim: {num_dim}\n"
                "Logit index 0 = target correct verb, index 1 = wrong verb"
            )
            after_text = base.build_50_sample_logs(
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
            print(after_text)

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
                    "num_dim": num_dim,
                    "mask_type": "hard_fixed",
                },
                "training": {
                    "seed": train_seed,
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
                    "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
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
