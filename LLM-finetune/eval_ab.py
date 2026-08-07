"""A/B 对比：微调前（基座）vs 微调后（猫娘 LoRA）
用法: python LLM-finetune/eval_ab.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "/kaggle/working/qwen05-lora"
PROMPTS = ["介绍一下你自己", "解释一下什么是神经网络", "你好"]

def run(model, tokenizer, prompt, device):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
print("=== 微调前（基座）===")
for p in PROMPTS:
    print(f"\nQ: {p}\nA: {run(base, tok, p, device)}")

ft = PeftModel.from_pretrained(base, ADAPTER).to(device)
print("\n=== 微调后（猫娘）===")
for p in PROMPTS:
    print(f"\nQ: {p}\nA: {run(ft, tok, p, device)}")