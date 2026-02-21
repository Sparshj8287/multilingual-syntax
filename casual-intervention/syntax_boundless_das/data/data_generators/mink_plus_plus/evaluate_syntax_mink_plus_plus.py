import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = (
        script_dir.parent / "templates" / "data" / "obj_rel_across_anim"
    )
    default_results_root = script_dir / "results"

    parser = argparse.ArgumentParser(
        description=(
            "Compute Min-K plus plus familiarity score and logit-difference syntax "
            "confidence for obj_rel_across_anim sentence pairs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing obj_rel_across_anim_pairs_{N}.jsonl files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
        help="Root output directory. Results are saved as results/<model_id>/results_{N}.jsonl.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-pt",
        help="Model checkpoint name or local path.",
    )
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        help="Enable instruction-style prompt wrapper around the prefix.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "Complete the following sentence with the correct form of the verb. "
            "Please answer in one word:\n"
            "Sentence: {sentence}\n"
            "Options: {options}\n"
            "Answer:\n"
        ),
        help=(
            "Prompt template used when --use-prompt is set. "
            "Supported placeholders: {sentence}, {options}, {correct_verb}, {incorrect_verb}."
        ),
    )
    parser.add_argument(
        "--k-ratio",
        type=float,
        default=0.2,
        help="Bottom-k ratio for Min-K plus plus (for example 0.2 for 20 percent).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--min-attractors",
        type=int,
        default=1,
        help="Minimum number of attractors (inclusive).",
    )
    parser.add_argument(
        "--max-attractors",
        type=int,
        default=6,
        help="Maximum number of attractors (inclusive).",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_model_id(model_name: str, use_prompt: bool = False) -> str:
    model_id = Path(model_name.rstrip("/")).name
    model_id = model_id.replace(" ", "_")
    if use_prompt:
        model_id = f"{model_id}-prompt"
    return model_id


def maybe_apply_prompt(
    prefix_text: str,
    correct_verb: str,
    incorrect_verb: str,
    use_prompt: bool,
    prompt_template: str,
) -> str:
    if not use_prompt:
        return prefix_text

    options = f"{correct_verb}, {incorrect_verb}"
    try:
        return prompt_template.format(
            sentence=prefix_text,
            options=options,
            correct_verb=correct_verb,
            incorrect_verb=incorrect_verb,
        )
    except KeyError as exc:
        raise ValueError(
            "Invalid --prompt-template placeholder. Supported keys: "
            "{sentence}, {options}, {correct_verb}, {incorrect_verb}."
        ) from exc


def isolate_prefix(sentence: str, target_verb: str) -> str:
    """
    Remove the target verb and trailing punctuation from sentence end.
    """
    sentence = sentence.strip()
    escaped = re.escape(target_verb.strip())
    pattern = re.compile(rf"\s+{escaped}\s*[^\w\s]*\s*$")
    match = pattern.search(sentence)


    if not match:
        if sentence.endswith(target_verb):
            return sentence[: -len(target_verb)].rstrip()
        raise ValueError(
            f"Could not isolate prefix. sentence='{sentence}', target_verb='{target_verb}'"
        )


    return sentence[: match.start()].rstrip()


def model_input_device(model: AutoModelForCausalLM) -> torch.device:
    return next(model.parameters()).device


def pick_candidate_token_id(
    tokenizer: AutoTokenizer, verb: str
) -> Tuple[int, str, bool]:
    """
    Prefer a single-token representation, trying spacing variants first.
    """
    base = verb.strip()
    surfaces: List[str] = []
    for candidate in (f" {base}", base, verb):
        if candidate and candidate not in surfaces:
            surfaces.append(candidate)

    best_surface = None
    best_ids: List[int] = []
    for surface in surfaces:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], surface, True
        if ids and (not best_ids or len(ids) < len(best_ids)):
            best_surface = surface
            best_ids = ids

    if not best_ids:
        raise ValueError(f"Could not tokenize candidate verb: '{verb}'")

    return best_ids[0], (best_surface or verb), False


def is_single_token_verb(
    tokenizer: AutoTokenizer, verb: str
) -> Tuple[bool, str, List[int]]:
    """
    Check whether a verb can be represented as a single token.
    """
    base = verb.strip()
    surfaces: List[str] = []
    for candidate in (f" {base}", base, verb):
        if candidate and candidate not in surfaces:
            surfaces.append(candidate)


    best_surface = base
    best_ids: List[int] = []
    for surface in surfaces:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return True, surface, ids
        if ids and (not best_ids or len(ids) < len(best_ids)):
            best_surface = surface
            best_ids = ids

    return False, best_surface, best_ids


def compute_mink_plus_plus(
    prefix_text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    k_ratio: float = 0.2,
) -> float:
    if not 0 < k_ratio <= 1:
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")

    encoded = tokenizer.encode(prefix_text)
    if len(encoded) < 2:
        raise ValueError(f"Prefix too short for Min-K%++: '{prefix_text}'")

    input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(
        model_input_device(model)
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits
    target_ids = input_ids[0, 1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)

    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    sigma = torch.clamp(sigma, min=1e-12)

    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    k_length = max(1, int(mink_plus.numel() * k_ratio))
    bottom_k = torch.sort(mink_plus).values[:k_length]
    return float(bottom_k.mean().item())


def compute_logit_diff(
    prefix_text: str,
    correct_verb: str,
    incorrect_verb: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> float:
    device = model_input_device(model)
    model_inputs = tokenizer(prefix_text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**model_inputs)

    next_token_logits = outputs.logits[0, -1, :]
    correct_id, _, _ = pick_candidate_token_id(tokenizer, correct_verb)
    incorrect_id, _, _ = pick_candidate_token_id(tokenizer, incorrect_verb)

    return float((next_token_logits[correct_id] - next_token_logits[incorrect_id]).item())


def process_example(
    row: Dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    k_ratio: float,
    use_prompt: bool,
    prompt_template: str,
) -> Dict[str, Any]:
    out = dict(row)

    base_prefix_raw = isolate_prefix(out["base_sentence"], out["MV_base"])
    source_prefix_raw = isolate_prefix(out["source_sentence"], out["MV_source"])
    base_prefix = maybe_apply_prompt(
        prefix_text=base_prefix_raw,
        correct_verb=out["MV_base"],
        incorrect_verb=out["MV_source"],
        use_prompt=use_prompt,
        prompt_template=prompt_template,
    )
    source_prefix = maybe_apply_prompt(
        prefix_text=source_prefix_raw,
        correct_verb=out["MV_source"],
        incorrect_verb=out["MV_base"],
        use_prompt=use_prompt,
        prompt_template=prompt_template,
    )

    out["base_mink_score"] = compute_mink_plus_plus(
        base_prefix, model=model, tokenizer=tokenizer, k_ratio=k_ratio
    )
    out["base_logit_diff"] = compute_logit_diff(
        base_prefix,
        correct_verb=out["MV_base"],
        incorrect_verb=out["MV_source"],
        model=model,
        tokenizer=tokenizer,
    )

    out["source_mink_score"] = compute_mink_plus_plus(
        source_prefix, model=model, tokenizer=tokenizer, k_ratio=k_ratio
    )
    out["source_logit_diff"] = compute_logit_diff(
        source_prefix,
        correct_verb=out["MV_source"],
        incorrect_verb=out["MV_base"],
        model=model,
        tokenizer=tokenizer,
    )
    out["use_prompt"] = bool(use_prompt)

    return out


def main() -> None:
    args = parse_args()
    if args.min_attractors > args.max_attractors:
        raise ValueError("--min-attractors must be <= --max-attractors")

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    model_id = infer_model_id(args.model, use_prompt=args.use_prompt)

    for attractor_n in range(args.min_attractors, args.max_attractors + 1):
        input_jsonl = args.data_dir / f"obj_rel_across_anim_pairs_{attractor_n}.jsonl"
        output_jsonl = args.results_root / model_id / f"results_{attractor_n}.jsonl"

        if not input_jsonl.exists():
            print(f"Skipping missing input: {input_jsonl}")
            continue

        rows = load_jsonl(input_jsonl)
        processed: List[Dict[str, Any]] = []
        discarded = 0
        discarded_verbs: List[str] = []
        for idx, row in enumerate(
            tqdm(
                rows,
                desc=f"Evaluating attractors={attractor_n}",
                total=len(rows),
            )
        ):
            try:
                base_ok, _, _ = is_single_token_verb(tokenizer, row["MV_base"])
                source_ok, _, _ = is_single_token_verb(tokenizer, row["MV_source"])
                if not base_ok or not source_ok:
                    discarded += 1
                    if not base_ok:
                        discarded_verbs.append(row["MV_base"])
                    if not source_ok:
                        discarded_verbs.append(row["MV_source"])
                    continue

                processed.append(
                    process_example(
                        row=row,
                        model=model,
                        tokenizer=tokenizer,
                        k_ratio=args.k_ratio,
                        use_prompt=args.use_prompt,
                        prompt_template=args.prompt_template,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed on attractor={attractor_n}, row index {idx}"
                ) from exc

        write_jsonl(output_jsonl, processed)
        print(
            f"Total examples: {len(rows)} | Discarded: {discarded} | "
            f"Kept: {len(processed)}"
        )
        if discarded_verbs:
            unique_verbs = sorted(set(discarded_verbs))
            print(f"Discarded verbs ({len(unique_verbs)}): {', '.join(unique_verbs)}")
        else:
            print("Discarded verbs: none")
        print(f"Saved {len(processed)} rows to: {output_jsonl}")


if __name__ == "__main__":
    main()
