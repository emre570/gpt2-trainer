# gpt2_turkish_fineweb_train.py

import re
import argparse
from itertools import islice
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 0. Argparse Setup
parser = argparse.ArgumentParser(description="Train GPT-2 from scratch on Turkish FineWeb data")
parser.add_argument("--dataset_size", type=int, default=100_000, help="Number of rows to load from FineWeb")
parser.add_argument("--num_cpu", type=int, default=4, help="Number of CPU cores for parallel processing")
parser.add_argument("--model_name", type=str, default="openai-community/gpt2", help="HuggingFace model config name")
args = parser.parse_args()

# 1. Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

# 2. Load & Preprocess Turkish FineWeb Dataset
print("Loading FineWeb data (streaming mode)...")
fw = load_dataset("HuggingFaceFW/fineweb-2", name="tur_Latn", split="train", streaming=True)

def clean_and_filter(example):
    if isinstance(example["text"], str):
        cleaned = re.sub(r"\s+", " ", example["text"]).strip()
        return {"text": cleaned}
    else:
        return {"text": ""}

cleaned_fw = fw.map(clean_and_filter)
filtered_fw = cleaned_fw.filter(lambda x: len(x["text"]) > 0)

sample_list = list(islice(filtered_fw, args.dataset_size))
ds = Dataset.from_list(sample_list)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize, batched=True, num_proc=args.num_cpu)
ds = ds.map(lambda x: {"label": x["input_ids"]}, num_proc=args.num_cpu)
ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 3. Model Config
config = AutoConfig.from_pretrained(args.model_name)
config.use_cache = False

model = AutoModelForCausalLM.from_config(config)
model.resize_token_embeddings(len(tokenizer))

# 4. Inference Logger
class InferenceLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        prompt = "Yapay zeka gelecekte "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_k=50,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"[Epoch {state.epoch:.2f}] Output: {decoded[:100]:<100}")

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-fineweb-tr",
    per_device_train_batch_size=112,
    gradient_accumulation_steps=1,
    num_train_epochs=100,
    logging_steps=1000,
    save_steps=1250,
    learning_rate=1e-4,
    fp16=True,
    save_total_limit=5,
    report_to="none",
    remove_unused_columns=False,
)

# 6. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    callbacks=[InferenceLoggerCallback()]
)

# 7. Start Training
trainer.train()
