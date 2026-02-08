import os
import sys

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: nltk. Install it before running this script."
    ) from exc


ANIMATE_LEXNAMES = {"noun.person", "noun.animal"}


def is_animate(lemma):
    lemma = lemma.lower()
    try:
        synsets = wn.synsets(lemma)
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        synsets = wn.synsets(lemma)

    for syn in synsets:
        if syn.lexname() in ANIMATE_LEXNAMES:
            return True
    return False


def process_wordnet(input_base_dir, output_base_dir):
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory not found at {input_base_dir}")
        return sys.exit(1)

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    total_animate = 0
    total_inanimate = 0
    total_files = 0

    for root, _, files in os.walk(input_base_dir):
        rel_dir = os.path.relpath(root, input_base_dir)
        output_dir = os.path.join(output_base_dir, rel_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in files:
            if filename.startswith("."):
                continue

            input_path = os.path.join(root, filename)
            animate_path = os.path.join(output_dir, f"animate_{filename}")
            inanimate_path = os.path.join(output_dir, f"inanimate_{filename}")

            file_animate = 0
            file_inanimate = 0

            with open(input_path, "r", encoding="utf-8") as infile, \
                    open(animate_path, "w", encoding="utf-8") as anim_out, \
                    open(inanimate_path, "w", encoding="utf-8") as inanim_out:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if not parts:
                        continue

                    lemma = parts[0]

                    if is_animate(lemma):
                        anim_out.write(f"{line}\n")
                        total_animate += 1
                        file_animate += 1
                    else:
                        inanim_out.write(f"{line}\n")
                        total_inanimate += 1
                        file_inanimate += 1

            total_files += 1
            print(
                f"{filename}: animate={file_animate}, inanimate={file_inanimate}"
            )

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Files processed: {total_files}")
    print(f"Total animate: {total_animate}")
    print(f"Total inanimate: {total_inanimate}")
    print(f"Output directory: {output_base_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "../eng/raw_data")
    output_base = os.path.join(script_dir, "../eng/wordnet_filtered")
    process_wordnet(input_path, output_base)
