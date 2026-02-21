import argparse
from typing import Dict, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

DEFAULT_PROMPT = "The competitor who the guys starve "
DEFAULT_CHOICE_A = "emerge"
DEFAULT_CHOICE_B = "emerges"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renormalized two-candidate next-word score for Gemma-3 IT."
    )
    parser.add_argument(
        "--ckpt",
        default="google/gemma-3-4b-pt",
        help="Model checkpoint name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt string to run.",
    )
    parser.add_argument(
        "--choice-a",
        default=DEFAULT_CHOICE_A,
        help="First candidate word.",
    )
    parser.add_argument(
        "--choice-b",
        default=DEFAULT_CHOICE_B,
        help="Second candidate word.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum number of new tokens to generate for inspection.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index to force (for example: 2 for cuda:2). Defaults to auto device mapping.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat templating and pass prompt directly.",
    )
    parser.add_argument(
        "--strict-single-token",
        action="store_true",
        help="Fail if a candidate is not representable as exactly one token.",
    )
    return parser.parse_args()


def resolve_device_map(cuda_device: Optional[int]) -> Union[str, Dict[str, str]]:
    if not torch.cuda.is_available():
        if cuda_device is not None:
            print("CUDA is not available; ignoring --cuda-device and running on CPU.")
        return {"": "cpu"}

    if cuda_device is None:
        return "auto"

    available = torch.cuda.device_count()
    if cuda_device < 0 or cuda_device >= available:
        raise ValueError(
            f"Invalid --cuda-device={cuda_device}. Available CUDA devices: 0..{available - 1}"
        )
    return {"": f"cuda:{cuda_device}"}


def build_model_inputs(
    tokenizer: AutoTokenizer, prompt: str, device: torch.device, use_chat_template: bool
) -> Tuple[dict, str]:
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not support apply_chat_template. "
                "Upgrade transformers/tokenizer or use --no-chat-template."
            )
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    return tokenizer(formatted_prompt, return_tensors="pt").to(device), formatted_prompt


def pick_candidate_token(
    tokenizer: AutoTokenizer, candidate: str, strict_single_token: bool
) -> Tuple[int, str, list[int], bool]:
    base = candidate.strip()
    surfaces = []
    for surface in (candidate, f" {base}", base):
        if surface and surface not in surfaces:
            surfaces.append(surface)

    tokenizations = []
    for surface in surfaces:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        tokenizations.append((surface, ids))
        if len(ids) == 1:
            return ids[0], surface, ids, True

    valid = [(surface, ids) for surface, ids in tokenizations if ids]
    if not valid:
        raise ValueError(f"Could not tokenize candidate: '{candidate}'")

    best_surface, best_ids = min(valid, key=lambda item: len(item[1]))
    if strict_single_token:
        details = ", ".join(f"{surface} -> {ids}" for surface, ids in tokenizations)
        raise ValueError(
            f"Candidate '{candidate}' is not a single token. Tried: {details}"
        )

    return best_ids[0], best_surface, best_ids, False


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    device_map = resolve_device_map(args.cuda_device)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.ckpt,
        device_map=device_map,
        torch_dtype=dtype,
    )
    model.eval()

    model_inputs, formatted_prompt = build_model_inputs(
        tokenizer, args.prompt, model.device, use_chat_template=not args.no_chat_template
    )
    input_len = model_inputs.input_ids.shape[-1]

    with torch.inference_mode():
        outputs = model(**model_inputs)

    next_token_logits = outputs.logits[0, -1, :]
    full_vocab_probs = torch.softmax(next_token_logits, dim=-1)

    a_id, a_surface, a_ids, a_exact = pick_candidate_token(
        tokenizer, args.choice_a, args.strict_single_token
    )
    b_id, b_surface, b_ids, b_exact = pick_candidate_token(
        tokenizer, args.choice_b, args.strict_single_token
    )

    pair_logits = torch.stack([next_token_logits[a_id], next_token_logits[b_id]])
    pair_probs = torch.softmax(pair_logits, dim=0)

    a_raw = full_vocab_probs[a_id].item()
    b_raw = full_vocab_probs[b_id].item()
    a_renorm = pair_probs[0].item()
    b_renorm = pair_probs[1].item()

    winner = args.choice_a if a_renorm > b_renorm else args.choice_b

    print("Prompt:\n" + args.prompt)
    if not args.no_chat_template:
        print("\nTemplated prompt:\n" + formatted_prompt)
    if torch.cuda.is_available():
        if args.cuda_device is None:
            print("\nCUDA device selection: auto")
        else:
            print(f"\nCUDA device selection: cuda:{args.cuda_device}")
    else:
        print("\nCUDA device selection: CPU")

    print("\nCandidate tokenization:")
    print(
        f"- {args.choice_a}: selected surface='{a_surface}', token_ids={a_ids}, exact_single_token={a_exact}"
    )
    print(
        f"- {args.choice_b}: selected surface='{b_surface}', token_ids={b_ids}, exact_single_token={b_exact}"
    )
    if not a_exact or not b_exact:
        print(
            "\nWarning: one or both candidates were not a single token. "
            "Renormalized score used the first token fallback."
        )

    print("\nRaw next-token probabilities (full vocabulary):")
    print(f"- P({args.choice_a}) = {a_raw:.100f}")
    print(f"- P({args.choice_b}) = {b_raw:.100f}")

    print("\nRenormalized over the two candidates only:")
    print(f"- P({args.choice_a} | {args.choice_a} vs {args.choice_b}) = {a_renorm:.6f}")
    print(f"- P({args.choice_b} | {args.choice_a} vs {args.choice_b}) = {b_renorm:.6f}")
    print(f"\nPreferred candidate: {winner}")

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0, input_len:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print("\nGreedy completion:\n" + completion)


if __name__ == "__main__":
    main()
