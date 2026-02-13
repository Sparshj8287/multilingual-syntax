import os
import json
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
ckpt = "google/gemma-3-12b-pt"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = Gemma3ForCausalLM.from_pretrained(
    ckpt,
    device_map="auto",
    dtype=torch.bfloat16,
)

test_sent = "the competitor who the guys starve " # No trailing space
# prompt = prefix + test_sent
prompt = test_sent

corret_word = "emerges"
wrong_word = "emerge"

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = model_inputs.input_ids.shape[-1]

with torch.inference_mode():
    generation = model(**model_inputs)

logits = generation.logits[0, -1, :]
probs = torch.softmax(logits, dim=-1)

correct_token_ids = tokenizer.encode(corret_word, add_special_tokens=False)
wrong_token_ids = tokenizer.encode(wrong_word, add_special_tokens=False)

correct_token_probs = [probs[i].item() for i in correct_token_ids]
wrong_token_probs = [probs[i].item() for i in wrong_token_ids]

correct_token_prob = sum(correct_token_probs)
wrong_token_prob = sum(wrong_token_probs)

print(f"{corret_word} token probability: {correct_token_prob:.20f} ({correct_token_prob*100:.2f}%)")
print(f"{wrong_word} token probability: {wrong_token_prob:.20f} ({wrong_token_prob*100:.2f}%)")

if correct_token_prob > wrong_token_prob:
    print(f"Correct word: {corret_word} is more likely to be generated than wrong word: {wrong_word}")
else:
    print(f"Wrong word: {wrong_word} is more likely to be generated than correct word: {corret_word}")

print("\n\n--------------------------------\n\n")
# Print the top 5 predictions
top_token_id = torch.argmax(probs).item()
top_token_word = tokenizer.decode(top_token_id)
print(f"Top token ID: {top_token_id}, Word: {top_token_word}")

top_k = torch.topk(probs, 5)
print("\nTop 5 Predictions:")
for i in range(5):
    tok_id = top_k.indices[i].item()
    tok_prob = top_k.values[i].item()
    tok_text = tokenizer.decode([tok_id])
    print(f"{i+1}. '{tok_text}' ({tok_prob:.4f})")

