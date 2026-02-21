import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

word = "articulates"


tokens = tokenizer.encode(word, return_tensors="pt", add_special_tokens=False)

print(tokens)



if len(tokens) > 1:
    print("Multiple tokens found")
else:
    print("Single token found")