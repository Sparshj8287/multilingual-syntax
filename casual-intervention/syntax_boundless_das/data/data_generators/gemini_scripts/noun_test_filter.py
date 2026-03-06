import argparse
import json
import os
import sys
import time

from google import genai
from google.genai import errors, types

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm isn't installed
    tqdm = None


PROMPT_TEMPLATE = """You are an expert Psycholinguist and Computational Linguist creating a strict Out-Of-Distribution (OOD) held-out test dataset for a Subject-Verb Agreement experiment. 

Your Goal: I will provide you with a raw, noisy batch of 50 nouns. You must act as a strict gatekeeper. You will DISCARD all common, easy, or invalid words, and KEEP ONLY the rare, highly academic, or morphologically irregular nouns.

Your Task: Process the raw list below. Evaluate each word against the strict selection criteria. If a word passes the filter, output it in the EXACT SAME JSONL format as requested, without adding any new fields.

### 1. Selection Rules (KEEP These)
To be selected, a noun MUST fall into one of these two categories:
* Category A: IRREGULAR PLURALS
  * Definition: Nouns that do not form their plural by simply adding "-s" or "-es". 
  * Examples: criterion/criteria, index/indices, ox/oxen, mouse/mice, thesis/theses.
* Category B: RARE / ACADEMIC / COMPLEX
  * Definition: Low-frequency, highly domain-specific, or multi-syllabic academic nouns. 

### 2. Rejection Rules (Strictly DISCARD These)
If a word fits any of these descriptions, SKIP IT ENTIRELY. Do not include it in the output.
* Common / High-Frequency: Everyday words (e.g., dog, car, teacher, student, manager). They belong in the training set, not the OOD test set.
* Ambiguous Plurals: Words where Singular and Plural are identical (e.g., sheep, aircraft). These ruin agreement experiments.
* Mass / Uncountable Nouns: Words that do not naturally pluralize (e.g., furniture, dust).
* Wrong POS / Invalid: Verbs, adjectives, proper nouns (names/places), or slang.

### 3. Output Formatting (CRITICAL)
* Output strictly in JSONL format. One valid JSON object per line.
* Do NOT add any new fields (no 'complexity' or 'reason' fields).
* Output EXACTLY these four keys: "lemma", "plural", "category", "tags".
* No markdown code blocks, no intro text, no conversational filler.

Example Output (Your Target):
{"lemma": "criterion", "plural": "criteria", "category": "INANIMATE", "tags": ["IS"]}
{"lemma": "oligarch", "plural": "oligarchs", "category": "ANIMATE", "tags": ["MS", "ES", "BS"]}
{"lemma": "matrix", "plural": "matrices", "category": "INANIMATE", "tags": ["IS"]}

Raw Word List to Process:
[PASTE YOUR BATCH OF 50 WORDS HERE]
"""


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(
        script_dir,
        "../../eng/raw_data_cleaned/noun/gemini_nouns.jsonl",
    )
    default_output = os.path.join(
        script_dir,
        "../../eng/raw_data_cleaned/noun/gemini_nouns_test.jsonl",
    )
    parser = argparse.ArgumentParser(description="Filter nouns into an OOD test set with Gemini.")
    parser.add_argument(
        "--input",
        default=default_input,
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of words per Gemini prompt.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between batches.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per batch on server errors.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=5.0,
        help="Base backoff seconds for retries.",
    )
    return parser.parse_args()


def load_lines(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
    lines = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                # Optionally validate it's JSONL
                try:
                    json.loads(line)
                    lines.append(line)
                except json.JSONDecodeError:
                    print(f"Warning: skipping invalid JSON line: {line}")
    return lines


def chunk_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_prompt(batch_lines):
    batch_text = "\n".join(batch_lines)
    return PROMPT_TEMPLATE.replace(
        "[PASTE YOUR BATCH OF 50 WORDS HERE]",
        batch_text,
    )


def extract_jsonl(text):
    output_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        output_lines.append(json.dumps(obj, ensure_ascii=False))
    return output_lines


def main():
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    lines = load_lines(args.input)
    if not lines:
        print("No input lines found.")
        return

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    batches = list(chunk_list(lines, args.batch_size))
    iterator = tqdm(batches, desc="Gemini batches") if tqdm else batches

    with open(args.output, "w", encoding="utf-8") as outfile:
        for batch in iterator:
            prompt = build_prompt(batch)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            response_text = ""
            attempt = 0
            while True:
                attempt += 1
                try:
                    for chunk in client.models.generate_content_stream(
                        model=args.model,
                        contents=contents,
                        config=config,
                    ):
                        if chunk.text:
                            response_text += chunk.text
                    break
                except errors.ServerError as exc:
                    if attempt > args.max_retries:
                        raise
                    wait_time = args.retry_backoff * (2 ** (attempt - 1))
                    print(
                        f"ServerError on batch (attempt {attempt}/{args.max_retries}). "
                        f"Retrying in {wait_time:.1f}s: {exc}"
                    )
                    time.sleep(wait_time)
                    response_text = ""
                except errors.APIError as exc:
                    if attempt > args.max_retries:
                        raise
                    wait_time = args.retry_backoff * (2 ** (attempt - 1))
                    print(
                        f"APIError on batch (attempt {attempt}/{args.max_retries}). "
                        f"Retrying in {wait_time:.1f}s: {exc}"
                    )
                    time.sleep(wait_time)
                    response_text = ""

            jsonl_lines = extract_jsonl(response_text)
            if not jsonl_lines:
                print("Warning: No valid JSONL lines parsed for batch.")
            else:
                outfile.write("\n".join(jsonl_lines))
                outfile.write("\n")
                outfile.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()