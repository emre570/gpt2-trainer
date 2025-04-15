## 🧠 GPT-2 Turkish Pretraining on FineWeb

> “I had free GPUs lying around, so I trained a GPT-2 from scratch on Turkish data.”

This repo provides a single-file implementation to train a GPT-2 model from scratch using the Turkish split of the HuggingFace FineWeb-2 dataset. It's minimal, practical, and does exactly what it says.

---

## 🚀 Features

* Uses HuggingFace's `Trainer` + `Accelerate` for multi-GPU + FP16 support
* Streams the dataset — no full download or RAM overload
* Tokenization is parallelized with `--num_cpu` option
* Each epoch prints a sample inference (`"Yapay zeka gelecekte..."`)
* Argparse interface for flexibility

---

## 🛠️ Setup

### 1. Create environment & install dependencies

```bash
python3 -m venv gpt2-train
./source/gpt2-train/bin/activate
pip install -r requirements.txt
```

### 2. Configure `accelerate`

```bash
accelerate config
```

Recommended answers:

* This machine
* Multi-GPU if you have more than one
* DeepSpeed: No
* Compute dtype: fp16
* Use same device for all processes: Yes

---

## 🧪 Run training

```bash
accelerate launch gpt2_turkish_fineweb_train.py
  --dataset_size 350000
  --num_cpu 8
  --model_name openai-community/gpt2
```

You can replace `--dataset_size` with up to 89M rows (that's the total available for Turkish in FineWeb-2).

---

## 📦 Output

Trained models are saved to:

```bash
/gpt2-fineweb-tr/
```

It can be pushed to the Hub if needed.

---

## 💬 Example output

```bash
[Epoch 5.00] Output: Yapay zeka gelecekte devlet politikalarında önemli bir rol oynayabilir.
```

---

## 🧃 Why?

Honestly? Just to prove it can be done.
