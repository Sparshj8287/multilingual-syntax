import os
import sys

import spacy

# Load the English model (Small is fast and sufficient for this)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print(
        "Error: spaCy model 'en_core_web_sm' not found. "
        "Install it with: python -m spacy download en_core_web_sm"
    )
    sys.exit(1)

# CONFIGURATION
INPUT_DIR = "raw_data"
OUTPUT_DIR = "raw_data_cleaned"

# Strict Rules: Only keep words that match these tags
# PROPN = Proper Noun (Names), NOUN = Common Noun
STRICT_POS_MAP = {
    "noun": {"NOUN", "PROPN"},
    "verb": {"VERB"},
    "adjective": {"ADJ"},
}

SAMPLE_DROPPED = 5
VERB_FORM_ORDER = [
    ("V;PST", "pst"),
    ("V;V.PTCP;PRS", "vptcp_prs"),
    ("V;V.PTCP;PST", "vptcp_pst"),
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_pos_from_rel_dir(rel_dir):
    if rel_dir == ".":
        return None
    return rel_dir.split(os.sep, 1)[0]


def parse_lemma(line):
    parts = line.split("\t")
    if parts:
        return parts[0].strip()
    return line.strip().split(" ", 1)[0]


def clean_file(input_path, output_path, file_type):
    clean_lines = []
    dropped_examples = []
    seen_lines = set()

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = [line.rstrip("\n") for line in infile if line.strip()]

    if not lines:
        open(output_path, "w", encoding="utf-8").close()
        return 0, 0, dropped_examples

    raw_words = [parse_lemma(line) for line in lines]

    for original_line, word, doc in zip(lines, raw_words, nlp.pipe(raw_words)):
        pos = doc[0].pos_ if doc else None
        is_valid = pos in STRICT_POS_MAP[file_type]

        # Extra "Common Sense" Filters
        if file_type == "noun" and word.endswith("ing"):
            is_valid = False

        if is_valid:
            if original_line in seen_lines:
                continue
            seen_lines.add(original_line)
            clean_lines.append(original_line)
        else:
            if len(dropped_examples) < SAMPLE_DROPPED:
                dropped_examples.append(f"{word} ({pos})")

    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(clean_lines))
        if clean_lines:
            outfile.write("\n")

    return len(clean_lines), len(lines) - len(clean_lines), dropped_examples


def process_directory(input_base, output_base):
    total_files = 0
    total_kept = 0
    total_dropped = 0

    for root, _, files in os.walk(input_base):
        rel_dir = os.path.relpath(root, input_base)
        file_type = get_pos_from_rel_dir(rel_dir)
        if file_type is None:
            continue
        if file_type not in STRICT_POS_MAP:
            continue

        output_dir = os.path.join(output_base, rel_dir)
        ensure_dir(output_dir)

        for filename in files:
            if filename.startswith("."):
                continue
            if not filename.endswith((".tsv", ".txt")):
                continue

            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_dir, filename)

            kept, dropped, examples = clean_file(
                input_path,
                output_path,
                file_type,
            )
            total_files += 1
            total_kept += kept
            total_dropped += dropped

            print(
                f"{os.path.join(rel_dir, filename)}: "
                f"kept={kept}, removed={dropped}"
            )
            if examples:
                print(f"  -> Examples Removed: {examples}")

    print("-" * 30)
    print("Cleaning complete.")
    print(f"Files processed: {total_files}")
    print(f"Total kept: {total_kept}")
    print(f"Total removed: {total_dropped}")
    print(f"Output directory: {output_base}")


def load_form_map(file_path):
    forms = {}
    if not os.path.exists(file_path):
        return forms
    with open(file_path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            lemma = parts[0].strip()
            form = parts[1].strip()
            if lemma and form and lemma not in forms:
                forms[lemma] = form
    return forms


def combine_verbs_for_prefix(verb_dir, prefix):
    prefix_str = f"{prefix}_" if prefix else ""
    base_filename = f"{prefix_str}V;PRS;3;SG.tsv"
    base_path = os.path.join(verb_dir, base_filename)
    if not os.path.exists(base_path):
        print(f"Skip combining verbs: missing {base_path}")
        return 0

    form_maps = {}
    for tag, label in VERB_FORM_ORDER:
        form_maps[label] = load_form_map(
            os.path.join(verb_dir, f"{prefix_str}{tag}.tsv")
        )

    combined_lines = []
    seen_lines = set()
    with open(base_path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            lemma = parts[0].strip() if parts else ""
            base_form = parts[1].strip() if len(parts) > 1 else ""
            if not lemma:
                continue

            fields = [lemma]
            forms = []
            if base_form:
                forms.append(base_form)

            for _, label in VERB_FORM_ORDER:
                form = form_maps[label].get(lemma)
                if form:
                    forms.append(form)

            seen = set()
            for form in forms:
                if form in seen:
                    continue
                seen.add(form)
                fields.append(form)

            fields.append("Verb")
            entry = "\t".join(fields)
            if entry in seen_lines:
                continue
            seen_lines.add(entry)
            combined_lines.append(entry)

    output_name = f"{prefix}_verb.tsv" if prefix else "verb.tsv"
    output_path = os.path.join(verb_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(combined_lines))
        if combined_lines:
            outfile.write("\n")

    print(f"Combined verbs written: {output_path}")
    return len(combined_lines)


def combine_verbs(output_base):
    verb_dir = os.path.join(output_base, "verb")
    if not os.path.exists(verb_dir):
        print(f"Skip combining verbs: missing {verb_dir}")
        return

    total = 0
    prefixes = []
    if os.path.exists(os.path.join(verb_dir, "animate_V;PRS;3;SG.tsv")):
        prefixes.extend(["animate", "inanimate"])
    elif os.path.exists(os.path.join(verb_dir, "V;PRS;3;SG.tsv")):
        prefixes.append("")

    if not prefixes:
        print("Skip combining verbs: no base verb files found.")
        return

    for prefix in prefixes:
        total += combine_verbs_for_prefix(verb_dir, prefix)

    keep_files = {"animate_verb.tsv", "inanimate_verb.tsv", "verb.tsv"}
    for filename in os.listdir(verb_dir):
        if filename in keep_files:
            continue
        file_path = os.path.join(verb_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("-" * 30)
    print("Verb combination complete.")
    print(f"Total combined verbs: {total}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "../eng", INPUT_DIR)
    output_path = os.path.join(script_dir, "../eng", OUTPUT_DIR)

    if not os.path.exists(input_path):
        print(f"Error: Input directory not found at {input_path}")
        sys.exit(1)

    process_directory(input_path, output_path)
    combine_verbs(output_path)
