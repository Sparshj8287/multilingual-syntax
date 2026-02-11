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


PROMPT_TEMPLATE = """
You are an expert Computational Linguist. Your task is to classify a raw list of verbs into strict syntactic categories (Slots) based on the **Recursive Grammar Templates** provided below.

**Context: The Template Architecture**
We are building a dataset using the following Python template rules. A verb's "Tag" is determined by its position and function in these specific structures.

**Reference Code (The Ground Truth):**

```python
# KEY LEGEND:
# MS = Main Subject (Human)   | IS = Inanimate Subject (Object)
# ES = Embedded Subject (Human) | BS = Base Subject (Thinker/Speaker)
# MV = Main Verb (Intransitive) | IV = Inanimate Verb (Intransitive)
# EV = Embedded Verb (Transitive)| BV = Base Verb (Mental/Speech)
# RMV = Reflexive Main Verb

self.rules = {
    # 1. Main Verb (MV) Logic:
    # Structure: [D, MS, ..., MV] -> "The author ... laughs."
    # Rule: MV must be an action performed by a Human (MS) that does NOT require an object (Intransitive).
    'simple_agrmt': (['D', 'MS', 'MV'], ...),

    # 2. Inanimate Verb (IV) Logic:
    # Structure: [D, IS, ..., IV] -> "The movie ... ends."
    # Rule: IV must be an action performed by an Object (IS) (Intransitive).
    'obj_rel_across_inanim': (['D', 'IS', ..., 'IV'], ...),

    # 3. Embedded Verb (EV) Logic:
    # Structure: [D, MS, C, D, ES, EV, MV] -> "The author that the guards [EV]..."
    # Meaning: "The guards [EV] the author."
    # Rule: EV must be Transitive and take a Human Object (MS).
    'obj_rel_across_anim': (..., 'EV', 'MV'),

    # 4. Base Verb (BV) Logic:
    # Structure: [D, BS, BV, D, MS, MV] -> "The banker [BV] that the pilot laughs."
    # Rule: BV must be a verb of Thinking/Saying (taking a Sentential Complement).
    'sent_comp': (['D', 'BS', 'BV', ...], ...),

    # 5. Reflexive Main Verb (RMV) Logic:
    # Structure: [D, MS, RMV, ANPHR] -> "The author [RMV] himself."
    # Rule: RMV must be a Transitive verb that a human does to themselves.
    'simple_reflexives': (['D', 'MS', 'RMV', 'ANPHR'], ...)
}

```

---

**Your Task:**
Process the raw verb list below. For each verb, extract the **Singular (3rd Person)** and **Plural (Base)** forms, and assign strict tags based on the logic above.

### **The Classification Rules (Test each verb against these Templates)**

**1. Tag: `MV` (Intransitive Human Action)**

* *Template Test:* `'simple_agrmt': ['D', 'MS', 'MV']`
* *Context:* "The **author** [VERBS]." (Must be complete without an object).
* *Examples:* *laughs, smiles, runs, sleeps, waits, speaks.*
* *Discard if:* It needs an object (e.g., *likes, puts*).

**2. Tag: `IV` (Intransitive Object Action)**

* *Template Test:* `'obj_rel_across_inanim': ['D', 'IS', ..., 'IV']`
* *Context:* "The **movie** [VERBS]."
* *Examples:* *ends, starts, falls, burns, breaks, shines.*

**3. Tag: `EV_ANIM` (Transitive Action on Human)**

* *Template Test:* `'obj_rel_across_anim': [..., 'MS', ..., 'EV']`
* *Context:* "The **author** that the guards [VERB]..." (Logic: Guards [VERB] Author).
* *Rule:* Must be Transitive. Target is **Human**.
* *Examples:* *likes, admires, hates, loves, hugs, criticizes, visits.*

**4. Tag: `EV_INAN` (Transitive Action on Object)**

* *Template Test:* `'obj_rel_across_inanim': [..., 'IS', ..., 'EV']`
* *Context:* "The **movie** that the guards [VERB]..." (Logic: Guards [VERB] Movie).
* *Rule:* Must be Transitive. Target is **Inanimate**.
* *Examples:* *watched, bought, read, wrote, broke, found.*

**5. Tag: `BV` (Mental/Speech Verb)**

* *Template Test:* `'sent_comp': ['D', 'BS', 'BV', ...]`
* *Context:* "The **banker** [VERBS] that..."
* *Examples:* *thought, knew, said, claimed, believed, argued, asserted.*

**6. Tag: `RMV` (Reflexive Action)**

* *Template Test:* `'simple_reflexives': [..., 'RMV', 'ANPHR']`
* *Context:* "The **author** [VERBS] himself."
* *Examples:* *hurt, injured, embarrassed, disguised, cut.*

---

**Output Format:**
Return valid **JSONL**.

* `sg`: 3rd Person Singular (e.g., "eats")
* `pl`: Base Form (e.g., "eat")
* `tags`: List of matching tags [`MV`, `IV`, `EV_ANIM`, `EV_INAN`, `BV`, `RMV`]

**Example Output:**

```json
{"sg": "laughs", "pl": "laugh", "tags": ["MV"]}
{"sg": "likes", "pl": "like", "tags": ["EV_ANIM", "EV_INAN"]}
{"sg": "reads", "pl": "read", "tags": ["EV_INAN"]}
{"sg": "thinks", "pl": "think", "tags": ["MV", "BV"]}
{"sg": "hurts", "pl": "hurt", "tags": ["RMV", "EV_ANIM", "EV_INAN"]}

```

Raw Verb List to Process:\n[PASTE YOUR LIST HERE]
"""


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(
        script_dir,
        "../../eng/raw_data_cleaned/verb/verb.tsv",
    )
    default_output = os.path.join(
        script_dir,
        "../../eng/raw_data_cleaned/verb/gemini_verbs.jsonl",
    )
    parser = argparse.ArgumentParser(description="Filter verbs with Gemini.")
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
        default=25,
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
    print(batch_text)    
    return PROMPT_TEMPLATE.replace(
        "Raw Verb List to Process:\n[PASTE YOUR LIST HERE]",
        f"Raw Verb List to Process:\n{batch_text}",
    )


def extract_jsonl(text):
    output_objects = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            output_objects.append(obj)
    return output_objects


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

    seen_verbs = set()

    with open(args.output, "w", encoding="utf-8") as outfile:
        for idx, batch in enumerate(iterator):
            time.sleep(3)

            batch_filtered = []
            for line in batch:
                parts = line.split("\t")
                if len(parts) >= 2:
                    batch_filtered.append("\t".join(parts[:2]))

            


            prompt = build_prompt(batch_filtered)
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]

            response_text = ""
            print(prompt)
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

            print(response_text)
            print("\n\n--------------------------------\n\n")

            json_objects = extract_jsonl(response_text)
            if not json_objects:
                print("Warning: No valid JSONL lines parsed for batch.")
            else:
                for obj in json_objects:
                    sg = obj.get("sg")
                    pl = obj.get("pl")
                    tags = obj.get("tags")
                    key = (sg, pl)
                    if key in seen_verbs:
                        continue
                    seen_verbs.add(key)
                    outfile.write(
                        json.dumps(
                            {"sg": sg, "pl": pl, "tags": tags},
                            ensure_ascii=False,
                        )
                    )
                    outfile.write("\n")
                outfile.flush()

            if args.sleep > 0:
                time.sleep(args.sleep)


    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()
