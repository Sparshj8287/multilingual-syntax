import json

def count_animate_and_inanimate(file_path):
    with open(file_path, 'r') as file:
        count_animate = 0
        count_inanimate = 0
        for line in file:
            data = json.loads(line)

            if data['category'] == 'ANIMATE':
                count_animate += 1
            else:
                count_inanimate += 1
    print(f"Count of animate: {count_animate}")
    print(f"Count of inanimate: {count_inanimate}")

def count_bs_tags(file_path):
    with open(file_path, 'r') as file:
        count_bs = 0
        for line in file:
            data = json.loads(line)
            if 'BS' in data['tags']:
                count_bs += 1
    print(f"Count of BS tags: {count_bs}")

if __name__ == "__main__":
    count_animate_and_inanimate("../../eng/raw_data_cleaned/noun/gemini_nouns.jsonl")
    count_bs_tags("../../eng/raw_data_cleaned/noun/gemini_nouns.jsonl")