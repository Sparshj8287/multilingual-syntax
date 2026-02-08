import os
import sys
import torch
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
INPUT_DIR = "raw_data"              # Where your .txt/.tsv files are
OUTPUT_DIR = "semantically_clustered"  # Where the split files go
MODEL_NAME = "BAAI/bge-m3"             # The "Gold Standard" model

# --- THE ANCHORS (The Rulers) ---
# We define "Animate" vs "Inanimate" differently for each POS tag.
# This ensures "Happy" (Adj) matches Animate, and "Rust" (Verb) matches Inanimate.
ANCHOR_CONFIG = {
    "noun": {
        "animate": "A living being, person, animal, creature, or human.",
        "inanimate": "A non-living object, physical artifact, abstract concept, place, or thing."
    },
    "verb": {
        # Check SUBJECT requirement: "Who does this?"
        "animate": "An action typically performed by a human, person, or animal.",
        "inanimate": "An action typically performed by a physical object, machine, or natural force."
    },
    "adjective": {
        # Check DESCRIPTION target: "What does this describe?"
        "animate": "An adjective describing a person's emotions, personality, behavior, or feelings.",
        "inanimate": "An adjective describing a physical object's material, shape, size, color, or state."
    }
}


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
        return parts[0]
    return line.strip().split(" ", 1)[0]


def cluster_file(
    input_path,
    output_dir,
    filename,
    model,
    anchor_embeddings,
):
    animate_path = os.path.join(output_dir, f"animate_{filename}")
    inanimate_path = os.path.join(output_dir, f"inanimate_{filename}")

    lines = []
    lemmas = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            lemma = parse_lemma(line)
            lines.append(line)
            lemmas.append(lemma)

    if not lines:
        open(animate_path, "w", encoding="utf-8").close()
        open(inanimate_path, "w", encoding="utf-8").close()
        return 0, 0

    embeddings = model.encode(
        lemmas,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sim = util.cos_sim(embeddings, anchor_embeddings)
    animate_mask = sim[:, 0] >= sim[:, 1]

    animate_count = 0
    inanimate_count = 0
    with open(animate_path, "w", encoding="utf-8") as anim_out, \
            open(inanimate_path, "w", encoding="utf-8") as inanim_out:
        for line, is_anim in zip(lines, animate_mask):
            if bool(is_anim):
                anim_out.write(f"{line}\n")
                animate_count += 1
            else:
                inanim_out.write(f"{line}\n")
                inanimate_count += 1

    return animate_count, inanimate_count


def process_directory(input_base, output_base, model):
    total_files = 0
    total_animate = 0
    total_inanimate = 0

    anchor_cache = {}

    for root, _, files in os.walk(input_base):
        rel_dir = os.path.relpath(root, input_base)
        pos = get_pos_from_rel_dir(rel_dir)

        if pos is not None and pos not in ANCHOR_CONFIG:
            continue

        output_dir = os.path.join(output_base, rel_dir)
        ensure_dir(output_dir)

        if pos not in anchor_cache:
            if pos is None:
                continue
            anchors = ANCHOR_CONFIG[pos]
            anchor_embeddings = model.encode(
                [anchors["animate"], anchors["inanimate"]],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            anchor_cache[pos] = anchor_embeddings

        anchor_embeddings = anchor_cache.get(pos)
        if anchor_embeddings is None:
            continue

        for filename in files:
            if filename.startswith("."):
                continue
            input_path = os.path.join(root, filename)
            animate_count, inanimate_count = cluster_file(
                input_path,
                output_dir,
                filename,
                model,
                anchor_embeddings,
            )
            total_files += 1
            total_animate += animate_count
            total_inanimate += inanimate_count
            print(
                f"{os.path.join(rel_dir, filename)}: "
                f"animate={animate_count}, inanimate={inanimate_count}"
            )

    print("-" * 30)
    print("Processing complete.")
    print(f"Files processed: {total_files}")
    print(f"Total animate: {total_animate}")
    print(f"Total inanimate: {total_inanimate}")
    print(f"Output directory: {output_base}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "../eng", INPUT_DIR)
    output_path = os.path.join(script_dir, "../eng", OUTPUT_DIR)

    if not os.path.exists(input_path):
        print(f"Error: Input directory not found at {input_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    process_directory(input_path, output_path, model)
