import argparse
import math
from typing import Dict, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

DEFAULT_PROMPT = """The representative that the finches encourage """



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run google/gemma-3-it on a prompt."
    )
    parser.add_argument(
        "--ckpt",
        default="google/gemma-3-12b-it",
        help="Model checkpoint name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt string to run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=None,
        help="Answer options to score (space-separated or comma-separated).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top next-token predictions to print.",
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
    return parser.parse_args()


def normalize_options(raw_options: list[str]) -> list[str]:
    if len(raw_options) == 1 and "," in raw_options[0]:
        return [opt.strip() for opt in raw_options[0].split(",") if opt.strip()]
    return raw_options


def options_from_prompt(prompt: str) -> list[str]:
    for line in prompt.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() == "options":
            items = [opt.strip() for opt in value.split(",") if opt.strip()]
            print(f"Options: {items}")
            if items:
                return items
    return []


def resolve_device_map(cuda_device: Optional[int]) -> Union[str, Dict[str, str]]:
    if not torch.cuda.is_available():
        if cuda_device is not None:
            print("CUDA is not available; ignoring --cuda-device and running on CPU.")
        return {"": "cpu"}

    if cuda_device is None:
        return "auto"

    available = torch.cuda.device_count()
    print(f"Available CUDA devices: {available}")
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


def compute_option_probability(
    model: Gemma3ForConditionalGeneration,
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
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = model_inputs["input_ids"].shape[-1]

    total_logprob = 0.0
    for idx, tok_id in enumerate(token_ids):
        total_logprob += log_probs[prompt_len + idx - 1, tok_id].item()
    return math.exp(total_logprob), token_ids


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.options:
        options = normalize_options(args.options)
    else:
        options = options_from_prompt(args.prompt)

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

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    option_results: list[tuple[str, float, list[int]]] = []
    for option in options:
        prob, token_ids = compute_option_probability(
            model, tokenizer, model_inputs, probs, option
        )
        option_results.append((option, prob, token_ids))

    best_option = max(option_results, key=lambda item: item[1])[0] if option_results else ""

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0, input_len:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

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
    print("\nCompletion:\n" + completion)
    print("\nFull text:\n" + full_text)
    print("\nOption probabilities:")
    for option, prob, token_ids in option_results:
        token_pieces = [tokenizer.decode([tok_id]) for tok_id in token_ids]
        token_info = " ".join(token_pieces) if token_pieces else "(no tokens)"
        print(f"- {option}: {prob:.8f} | tokens: {token_info}")

    if best_option:
        print(f"\nMost likely option: {best_option}")

    print("\n--------------------------------\n")
    top_k = min(args.top_k, probs.shape[0])
    top_k_vals, top_k_ids = torch.topk(probs, top_k)
    print(f"Top {top_k} Next-Token Predictions:")
    for rank, (tok_id, tok_prob) in enumerate(
        zip(top_k_ids.tolist(), top_k_vals.tolist()), start=1
    ):
        tok_text = tokenizer.decode([tok_id])
        print(f"{rank}. '{tok_text}' ({tok_prob:.4f})")


if __name__ == "__main__":
    main()
