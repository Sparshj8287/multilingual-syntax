import os
import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import glob

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_probabilities(model, tokenizer, prefix, target_word):
    # The prefix already includes the trailing space as requested.
    target_word = target_word.strip()


    
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
    # Get the token ID for the target word without prepending a space,
    # because the space is already at the end of the prefix.
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    
    if not target_ids:
        return 0.0, "N/A", False

    target_id = target_ids[0]
    target_prob = probs[target_id].item()
    
    max_prob_id = torch.argmax(probs).item()
    max_prob_word = tokenizer.decode([max_prob_id])
    
    return target_prob, max_prob_word, max_prob_id == target_id

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    
    config = load_config(config_path)
    models = config['models']
    data_dir = config['data_dir']
    

    data_path = os.path.abspath(os.path.join(script_dir, data_dir))
    
    jsonl_files = glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)
    
    if not jsonl_files:
        print(f"No JSONL files found in {data_path}")
        return

    for model_name in models:
        print(f"\n" + "="*50)
        print(f"Testing model: {model_name}")
        print("="*50)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        model_dir_name = model_name.replace("/", "_")
        output_model_dir = os.path.join(script_dir, model_dir_name)
        os.makedirs(output_model_dir, exist_ok=True)
        
        for jsonl_file in jsonl_files:

            rel_path = os.path.relpath(jsonl_file, data_path)
            file_basename = rel_path.replace(os.sep, "_").replace(".jsonl", ".txt")
            output_file_path = os.path.join(output_model_dir, file_basename)
            
            print(f"Processing {rel_path}...")
            
            base_correct = 0
            source_correct = 0
            total = 0
            
            results = []
            
            with open(jsonl_file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f"Entries"):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    base_sent = data['base_sentence']
                    source_sent = data['source_sentence']
                    mv_base = data['MV_base']
                    mv_source = data['MV_source']
                    
                    base_idx = base_sent.rfind(mv_base)
                    source_idx = source_sent.rfind(mv_source)

                    # print(base_idx, source_idx)
                    # print("\n\n-------------------------------------------------\n\n")
                    # print(base_sent)
                    # print(source_sent)
                    # print("\n\n-------------------------------------------------\n\n")
                    
                    if base_idx == -1 or source_idx == -1:
                        base_prefix = base_sent
                        source_prefix = source_sent
                    else:
                        base_prefix = base_sent[:base_idx]
                        source_prefix = source_sent[:source_idx]

                    # print(base_prefix, source_prefix)
                    # print("\n\n-------------------------------------------------\n\n")
                    
                    base_prob, base_max_word, base_is_correct = get_probabilities(model, tokenizer, base_prefix, mv_base)
                    source_prob, source_max_word, source_is_correct = get_probabilities(model, tokenizer, source_prefix, mv_source)
                    
                    if base_is_correct: base_correct += 1
                    if source_is_correct: source_correct += 1
                    total += 1
                    
                    results.append({
                        "base_prefix": base_prefix,
                        "base_target": mv_base,
                        "base_prob": base_prob,
                        "base_max_word": base_max_word,
                        "source_prefix": source_prefix,
                        "source_target": mv_source,
                        "source_prob": source_prob,
                        "source_max_word": source_max_word
                    })
            
            accuracy_base = base_correct / total if total > 0 else 0
            accuracy_source = source_correct / total if total > 0 else 0
            
            with open(output_file_path, 'w') as out_f:
                out_f.write(f"Model: {model_name}\n")
                out_f.write(f"File: {rel_path}\n")
                out_f.write(f"Base Sentence Overall Accuracy: {accuracy_base:.4f} ({base_correct}/{total})\n")
                out_f.write(f"Source Sentence Overall Accuracy: {accuracy_source:.4f} ({source_correct}/{total})\n")
                out_f.write("-" * 80 + "\n")
                
                for i, res in enumerate(results):
                    out_f.write(f"Entry {i+1}:\n")
                    out_f.write(f"  Base Prefix: {res['base_prefix']}\n")
                    out_f.write(f"  Base Target: {res['base_target']}\n")
                    out_f.write(f"  Base Max Prob Word: '{res['base_max_word']}'\n")
                    out_f.write(f"  Base Target Prob: {res['base_prob']:.6f}\n")
                    out_f.write(f"  Source Prefix: {res['source_prefix']}\n")
                    out_f.write(f"  Source Target: {res['source_target']}\n")
                    out_f.write(f"  Source Max Prob Word: '{res['source_max_word']}'\n")
                    out_f.write(f"  Source Target Prob: {res['source_prob']:.6f}\n")
                    out_f.write("\n")
            
            print(f"Results saved to {output_file_path}")
            print(f"Accuracies - Base: {accuracy_base:.4f}, Source: {accuracy_source:.4f}")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
