import json
import os

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_verbs_path = os.path.join(script_dir, "../../eng/raw_data_cleaned/verb/gemini_verbs.jsonl")
    test_verbs_path = os.path.join(script_dir, "../../eng/raw_data_cleaned/verb/gemini_verbs_test.jsonl")
    train_verbs_path = os.path.join(script_dir, "../../eng/raw_data_cleaned/verb/gemini_verbs_train.jsonl")
    
    print("Loading all verbs...")
    all_verbs = load_jsonl(all_verbs_path)
    print(f"Loaded {len(all_verbs)} verbs.")
    
    print("Loading test verbs...")
    test_verbs = load_jsonl(test_verbs_path)
    print(f"Loaded {len(test_verbs)} test verbs.")
    
    # Extract lemmas from test set for easy subtraction
    test_lemmas = {item["sg"] for item in test_verbs}
    
    print("Filtering train verbs...")
    train_verbs = [item for item in all_verbs if item["sg"] not in test_lemmas]
    print(f"Found {len(train_verbs)} train verbs after subtraction.")
    
    print(f"Saving train verbs to {train_verbs_path}...")
    with open(train_verbs_path, 'w', encoding='utf-8') as f:
        for item in train_verbs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("Done!")

if __name__ == "__main__":
    main()
