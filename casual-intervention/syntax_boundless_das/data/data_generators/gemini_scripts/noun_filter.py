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


PROMPT_TEMPLATE = """You are an expert Computational Linguist creating a high-quality dataset for a Subject-Verb Agreement experiment (similar to Marvin & Linzen, 2018). Your goal is to filter a raw, noisy list of words and classify them into strict semantic categories with specific template tags.

Your Task: Process the raw list of words below. For each word:

Clean it: Extract the strict Lemma (Singular) and Form (Plural). If only one is provided, generate the missing form.

Classify it: Assign it to strictly one of two categories (ANIMATE or INANIMATE).

Tag it: Assign the specific template tags (MS, ES, BS, IS) based on the logic below.

1. Classification Rules (The Logic)
You must classify every valid noun into exactly one of these two categories:

Category A (a): ANIMATE (Human / Profession)

Definition: Humans, professions, family roles, or distinct groups of people capable of communication and complex thought.

Template Tags: ["MS", "ES", "BS"]

Reasoning: Humans can be the Main Subject (MS), Embedded Subject (ES), and the "Thinker/Speaker" (Base Subject BS) for sentential complements (e.g., "The senator claimed that...").

Category A (b): ANIMATE (Animal)

Definition: Living animals that are not humans.

Template Tags: ["MS", "ES"]

Reasoning: Animals can be subjects, but generally cannot function as the "Base Subject" (BS) for verbs like claimed, asserted, wrote, said.

Category B: INANIMATE NOUNS

Definition: Non-living physical objects, artifacts, locations, abstract concepts, or natural phenomena.

Template Tags: Assign ["IS"].

Reasoning: Inanimate nouns serve as the Inanimate Subject (IS). They generally cannot be Base Subjects because they cannot "think" or "say" things.

Examples: mile, road, idea, movie, rifle, table, wind.

2. Filtering Rules (Strictly Discard These)
If a word belongs to any of the following categories, SKIP IT ENTIRELY. Do not include it in the output.

Wrong POS: Verbs (e.g., eat, tempting), Adjectives (chaotic, focal), or Adverbs.

Proper Nouns: Specific names of people or places (e.g., Daniel, London, Eden's, Soviet). We only want Common Nouns.

Slang / Offensive: Informal, rude, or highly colloquial terms (e.g., psycho, chick, motherfucker). Keep the tone formal and standard.

Ambiguous Plurals: Words where the Singular and Plural forms are identical (e.g., sheep, fish, moose). These ruin agreement experiments.

3. formatting Requirements
Input Handling: The input list is messy (e.g., mile (miles) N; PL). You must extract the clean Lemma (Singular) and Form (Plural).

Auto-Inflection: If the input only gives the singular, you must generate the correct plural form.

Output Format: strict JSONL. One valid JSON object per line. No markdown code blocks, no intro text.

Example Output (Your Target):

JSON
{"lemma": "mile", "plural": "miles", "category": "INANIMATE", "tags": ["IS"]}
{"lemma": "manager", "plural": "managers", "category": "ANIMATE", "tags": ["MS", "ES", "BS"]}
{"lemma": "mouse", "plural": "mice", "category": "ANIMATE", "tags": ["MS", "ES"]}
{"lemma": "cat", "plural": "cats", "category": "ANIMATE", "tags": ["MS", "ES"]}
Raw Word List to Process: [PASTE YOUR BATCH OF WORDS HERE]
"""


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(
        script_dir,
        "../../eng/raw_data_cleaned/noun/N;PL.tsv",
    )
    default_output = os.path.join(
        script_dir,
        "../../eng/raw_data_cleaned/noun/gemini_nouns.jsonl",
    )
    parser = argparse.ArgumentParser(description="Filter nouns with Gemini.")
    parser.add_argument(
        "--input",
        default=default_input,
        help="Path to input TSV file.",
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
        default="gemini-3-pro-preview",
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
    with open(input_path, "r", encoding="utf-8") as infile:
        return [line.strip() for line in infile if line.strip()]


def chunk_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_prompt(batch_lines):
    batch_text = "\n".join(batch_lines)
    return PROMPT_TEMPLATE.replace(
        "Raw Word List to Process: [PASTE YOUR BATCH OF WORDS HERE]",
        f"Raw Word List to Process:\n{batch_text}",
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