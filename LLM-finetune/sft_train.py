"""猫娘人设 LoRA 微调（标准 transformers + peft，T4 兼容 fp16）
用法: python LLM-finetune/sft_train.py
换模型只改 MODEL 一行。
"""
import json, torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "/kaggle/working/qwen05-lora"
DATA = "/kaggle/working/kaggle/LLM-finetune/persona.json"

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
    return f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}"
ds = Dataset.from_list([{"text": fmt(x)} for x in raw])

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=True,
    ),
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
trainer.train()
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("训练完成！adapter 在", OUT)