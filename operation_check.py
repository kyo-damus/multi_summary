import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "The highest mountain in the world is"

model_name = "gpt2"  


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

print(f"プロンプト: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response_text)