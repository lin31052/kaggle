"""猫娘人设 LoRA 微调（标准 transformers + peft，T4 兼容 fp16）
改进：只对 Response 部分计算 loss（指令前缀 mask 为 -100），epochs=8
用法: python LLM-finetune/sft_train.py
换模型只改 MODEL 一行。
"""
import json, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "/kaggle/working/qwen05-lora"
DATA = "/kaggle/working/kaggle/LLM-finetune/persona.json"
RESPONSE_TEMPLATE = "### Response:"

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0,
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

with open(DATA) as f:
    raw = json.load(f)
def fmt(x):
    return f"### Instruction:\n{x['instruction']}\n\n{RESPONSE_TEMPLATE}\n{x['output']}"
ds = Dataset.from_list([{"text": fmt(x)} for x in raw])

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# 只对 Response 部分算 loss 的 collator（指令前缀 mask 为 -100）
class CompletionCollator:
    def __init__(self, tokenizer, template=RESPONSE_TEMPLATE):
        self.tokenizer = tokenizer
        self.template_ids = tokenizer(template, add_special_tokens=False).input_ids

    def __call__(self, features):
        batch = {k: torch.tensor([f[k] for f in features]) for k in ("input_ids", "attention_mask")}
        labels = batch["input_ids"].clone()
        labels[:] = -100
        tlen = len(self.template_ids)
        for i, ids in enumerate(batch["input_ids"]):
            for j in range(len(ids) - tlen + 1):
                if ids[j:j + tlen].tolist() == self.template_ids:
                    labels[i, j:] = ids[j:]
                    break
        batch["labels"] = labels
        return batch

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=8,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
    ),
    train_dataset=tokenized,
    data_collator=CompletionCollator(tokenizer),
)
trainer.train()
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("训练完成！adapter 在", OUT)