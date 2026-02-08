import os
import sys

def sanitize_directory(input_dir):
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found at {input_dir}")
        sys.exit(1)

    print(f"Scanning directory: {input_dir}")

    total_removed = 0
    files_changed = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith((".tsv", ".txt")):
                continue

            file_path = os.path.join(root, filename)

            kept_lines = []
            removed_in_file = 0

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) < 3:
                        kept_lines.append(line)
                        continue

                    lemma = parts[0]
                    if '*' in lemma:
                        removed_in_file += 1
                        continue

                    kept_lines.append(line)

            if removed_in_file > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(kept_lines) + "\n")

                files_changed += 1
                total_removed += removed_in_file

    print("-" * 30)
    print("Sanitization complete.")
    print(f"Files modified: {files_changed}")
    print(f"Total lines removed: {total_removed}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "../eng/semantically_clustered")
    sanitize_directory(target_dir)