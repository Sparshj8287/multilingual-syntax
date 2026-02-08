import os
import sys
from wordfreq import zipf_frequency

language = "eng"

if language == "eng":
    lang_code ="en"

def category_for_features(features):
    head = features.split(";", 1)[0]
    if head == "V":
        return "verb"
    if head == "N":
        return "noun"
    if head == "ADJ":
        return "adjective"
    return None


def process_unimorph_with_freq(input_file, output_base_dir, min_freq=3.5):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    raw_data_dir = os.path.join(output_base_dir, "raw_data")
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        print(f"Created directory: {raw_data_dir}")

    category_dirs = {
        "verb": os.path.join(raw_data_dir, "verb"),
        "noun": os.path.join(raw_data_dir, "noun"),
        "adjective": os.path.join(raw_data_dir, "adjective"),
    }
    for directory in category_dirs.values():
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Dictionary to hold file handles for different features
    feature_files = {}

    print(f"Reading from: {input_file}")

    count_processed = 0
    count_skipped = 0
    count_unknown = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                lemma = parts[0]
                form = parts[1]
                features = parts[2]
                
                # Filter using wordfreq
                if zipf_frequency(lemma, lang_code) < min_freq:
                    count_skipped += 1
                    continue

                if features == "N;PL" and (form == "countable" or form == "uncountable"):
                    count_skipped += 1
                    continue
                
                if features =="N;SG":
                    count_skipped += 1
                    continue

                if features == "V;NFIN;IMP+SBJV":
                    count_skipped += 1
                    continue

                category = category_for_features(features)
                if category is None:
                    count_unknown += 1
                    continue

                if features not in feature_files:
                    feature_path = os.path.join(category_dirs[category], f"{features}.tsv")
                    feature_files[features] = open(feature_path, "w", encoding="utf-8")

                feature_files[features].write(f"{line}\n")
                count_processed += 1

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return sys.exit(1)
    finally:
        for handle in feature_files.values():
            handle.close()

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Words kept (freq > 0): {count_processed}")
    print(f"Words filtered out: {count_skipped}")
    if count_unknown:
        print(f"Words skipped (unknown features): {count_unknown}")
    print(f"Total files created in {raw_data_dir}: {len(feature_files)}")

if __name__ == "__main__":
    # Setup paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input: ../eng/eng
    input_path = os.path.join(script_dir, f"../{language}/{language}")
    
    # Output Base: ../eng/ (raw_data will be created inside)
    output_base = os.path.join(script_dir, f"../{language}")
    
    process_unimorph_with_freq(input_path, output_base)
