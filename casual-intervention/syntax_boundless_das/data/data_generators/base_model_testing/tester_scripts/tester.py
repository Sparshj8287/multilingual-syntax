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

prompt = "The competitor who the guys starve "
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = model_inputs.input_ids.shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded= tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)