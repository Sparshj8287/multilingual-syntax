import os
import json
from pathlib import Path

# Base directory paths
BASE_DIR = Path("/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/data/data_generators/templates/data/obj_rel_across_anim")
NEW_DIR = Path("/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/data/data_generators/templates/data/obj_rel_across_anim_2")

def process_file(file_path, new_file_path):
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'r') as f_in, open(new_file_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            
            # The bag-of-words hypothesis:
            # For 1 attractor, there is no change.
            # From 2 attractors onwards, the verb is predicted to agree with the plural/singular majority 
            # or due to the presence of attractors, the hypothesis predicts a change.
            # We swap the verbs for NUA >= 2.
            if data.get("NUA", 1) >= 2:
                # Replace the verb at the end of the sentence
                old_base_suffix = " " + data["MV_base"] + "."
                new_base_suffix = " " + data["MV_source"] + "."
                
                old_source_suffix = " " + data["MV_source"] + "."
                new_source_suffix = " " + data["MV_base"] + "."
                
                # Perform the replacements
                data["base_sentence"] = data["base_sentence"].replace(old_base_suffix, new_base_suffix)
                data["source_sentence"] = data["source_sentence"].replace(old_source_suffix, new_source_suffix)
                
                # Swap MV_base and MV_source in the dictionary so it matches the newly constructed sentences
                data["MV_base"], data["MV_source"] = data["MV_source"], data["MV_base"]
            
            # Write out the modified JSON line
            f_out.write(json.dumps(data) + '\n')

def main():
    print(f"Reading from {BASE_DIR}")
    print(f"Writing to {NEW_DIR}")
    
    for split in ['train', 'test']:
        split_dir = BASE_DIR / split / 'bow'
        if not split_dir.exists():
            print(f"Directory {split_dir} does not exist, skipping.")
            continue
            
        for file_name in os.listdir(split_dir):
            if file_name.endswith('.jsonl'):
                new_file_name = file_name.replace('obj_rel_across_anim', 'obj_rel_across_anim_2')
                file_path = split_dir / file_name
                new_file_path = NEW_DIR / split / 'bow' / new_file_name
                print(f"Processing {file_path} -> {new_file_path}")
                process_file(file_path, new_file_path)

if __name__ == "__main__":
    main()
