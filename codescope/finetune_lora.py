"""
finetune_lora.py — CodeScope LoRA Fine-tuning Script
======================================================
Fine-tunes a small causal LM (Phi-2 by default) on our code Q&A dataset
using PEFT LoRA. Designed to run on a free Colab T4 GPU (~15 min).

Usage:
    pip install peft transformers datasets accelerate bitsandbytes
    python finetune_lora.py

After training, the adapter is saved to ./lora_adapter/
The base model + adapter are then used for evaluation in evaluate.py
"""

import os, json, time
from pathlib import Path

# ── Try importing heavy deps; print install hint if missing ──────────────────
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        get_peft_model, LoraConfig, TaskType,
        PeftModel, prepare_model_for_kbit_training,
    )
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# ── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL   = "microsoft/phi-2"          # ~2.7B params, fits on free Colab T4
ADAPTER_DIR  = "./lora_adapter"
DATASET_PATH = "./eval_dataset.json"

LORA_CONFIG = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = 8,          # LoRA rank — low rank = fewer params, fast training
    lora_alpha   = 16,         # scaling factor (alpha/r = 2 is a common ratio)
    target_modules = ["q_proj", "v_proj"],   # which attention weights to adapt
    lora_dropout = 0.05,
    bias         = "none",
) if HAS_DEPS else None

TRAIN_ARGS = dict(
    output_dir          = "./lora_checkpoints",
    num_train_epochs    = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate       = 2e-4,
    fp16                = True,
    logging_steps       = 10,
    save_strategy       = "epoch",
    warmup_ratio        = 0.05,
    lr_scheduler_type   = "cosine",
    report_to           = "none",
)

# ── Prompt template ───────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "### Task: You are a codebase assistant. Answer questions about the code.\n"
    "### Question: {question}\n"
    "### Answer: {answer}"
)

def format_sample(item):
    """Converts a Q&A pair into a single training string."""
    return PROMPT_TEMPLATE.format(
        question=item["question"],
        answer=item["reference_answer"],
    )

# ── Data loading ──────────────────────────────────────────────────────────────

def load_train_data(path: str):
    with open(path) as f:
        data = json.load(f)
    # Use train + val splits for fine-tuning
    samples = data["train"] + data["val"]
    texts   = [format_sample(s) for s in samples]
    return Dataset.from_dict({"text": texts})

# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize(dataset, tokenizer, max_length=256):
    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return dataset.map(_tok, batched=True, remove_columns=["text"])

# ── Main fine-tuning routine ──────────────────────────────────────────────────

def run_finetune():
    if not HAS_DEPS:
        print("[!] Missing dependencies. Install with:")
        print("    pip install peft transformers datasets accelerate bitsandbytes")
        return

    print("=" * 60)
    print("  CodeScope — LoRA Fine-tuning")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  LoRA rank  : r={LORA_CONFIG.r}, alpha={LORA_CONFIG.lora_alpha}")
    print(f"  Targets    : {LORA_CONFIG.target_modules}")
    print("=" * 60)

    # 1. Load tokenizer & model
    print("\n[1/5] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 2. Attach LoRA adapter
    print("[2/5] Attaching LoRA adapter...")
    model = get_peft_model(model, LORA_CONFIG)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"      Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # 3. Prepare dataset
    print("[3/5] Loading & tokenizing dataset...")
    raw_ds  = load_train_data(DATASET_PATH)
    tok_ds  = tokenize(raw_ds, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 4. Train
    print("[4/5] Training...")
    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAIN_ARGS),
        train_dataset=tok_ds,
        data_collator=collator,
    )
    trainer.train()
    elapsed = time.time() - t0
    print(f"      Training done in {elapsed/60:.1f} min")

    # 5. Save adapter
    print("[5/5] Saving LoRA adapter...")
    Path(ADAPTER_DIR).mkdir(exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Save training summary
    summary = {
        "base_model": BASE_MODEL,
        "lora_rank": LORA_CONFIG.r,
        "lora_alpha": LORA_CONFIG.lora_alpha,
        "target_modules": list(LORA_CONFIG.target_modules),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "epochs": TRAIN_ARGS["num_train_epochs"],
        "learning_rate": TRAIN_ARGS["learning_rate"],
        "train_samples": len(raw_ds),
        "training_minutes": round(elapsed / 60, 1),
        "adapter_dir": ADAPTER_DIR,
    }
    with open("lora_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Adapter saved to: {ADAPTER_DIR}")
    print(f"   Training summary : lora_training_summary.json")
    print("\n   Run next: python evaluate.py")


# ── Inference helper (used by evaluate.py) ────────────────────────────────────

def load_finetuned_model(adapter_dir: str = ADAPTER_DIR):
    """Load base model + LoRA adapter for inference."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base      = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 150) -> str:
    """Generate an answer using the fine-tuned model."""
    prompt = (
        "### Task: You are a codebase assistant. Answer questions about the code.\n"
        f"### Question: {question}\n"
        "### Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the part after "### Answer:"
    return decoded.split("### Answer:")[-1].strip()


if __name__ == "__main__":
    run_finetune()
