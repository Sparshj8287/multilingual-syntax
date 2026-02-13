import argparse
import math
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

DEFAULT_PROMPT = """Fill in the blank with the grammatically correct verb form. Output the correct word and do not add any other text:
Sentence: The competitor who the guy starves _____
Options: emerge, emerges
Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run google/gemma-3-it on a prompt."
    )
    parser.add_argument(
        "--ckpt",
        default="google/gemma-3-1b-it",
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
        default=32,
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


def compute_option_probability(
    model: Gemma3ForCausalLM,
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
    model = Gemma3ForCausalLM.from_pretrained(
        args.ckpt,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    model_inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
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
