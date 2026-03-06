import json
import os 
from typing import List, Dict
from transformers import AutoTokenizer

MODELS_TO_USE = {
    # "gemma-3-12b-pt" : "google/gemma-3-12b-pt",
    # "llama-3.1-8b" : "/home/models/Llama-3.1-8B",
    "qwen-3-8b" : "Qwen/Qwen3-8B-Base",
    # "olmo-3-7b" : "allenai/Olmo-3-1025-7B"
}

VERB_FILE_PATH = "/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/data/eng/raw_data_cleaned/verb/gemini_verbs.jsonl"

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODELS_TO_USE[model_name])
    return tokenizer

def load_verb_data(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def filter_verbs(verbs: List[Dict], tokenizer: AutoTokenizer) -> List[Dict]:
    filtered_verbs = []

    for i in verbs:

        sg = i["sg"]
        pl = i["pl"]
        sg_tokens=tokenizer.encode(sg,add_special_tokens=False)
        pl_tokens=tokenizer.encode(pl,add_special_tokens=False)

        if len(sg_tokens) == 1 and len(pl_tokens) == 1:
            filtered_verbs.append(i)


    return filtered_verbs


def main():
    verbs = load_verb_data(VERB_FILE_PATH)
    temp_verbs = verbs.copy()
    print(f"Loaded {len(temp_verbs)} verbs")
    for model_name in MODELS_TO_USE:

        tokenizer = load_tokenizer(model_name)
        temp_verbs = filter_verbs(temp_verbs, tokenizer)
        print(f"Filtered {len(temp_verbs)} verbs for {model_name}")

    print(f"Filtered {len(temp_verbs)} verbs for all models")

    with open(f"/home/sparsh/projects/multilingual-syntax/casual-intervention/syntax_boundless_das/data/eng/raw_data_cleaned/verb/filtered_verbs.jsonl", "w") as f:
        for verb in temp_verbs:
            f.write(json.dumps(verb) + "\n")


if __name__ == "__main__":
    main()