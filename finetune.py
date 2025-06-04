import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ─────────────────────────────────────────────────────────────
# 1.  helpers
# ─────────────────────────────────────────────────────────────
def check_cuda() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"→ Device      : {dev}")
    if dev.type == "cuda":
        print(f"→ GPU name    : {torch.cuda.get_device_name(0)}")
        print(f"→ GPU mem (MB): {torch.cuda.get_device_properties(0).total_memory/1e6:,.0f}")
    return dev


def compute_token_accuracy(eval_pred):
    """Simple token‑level accuracy, ignoring padded label ids (‑100)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask   = labels != -100
    total  = mask.sum()
    if total == 0:
        return {"token_accuracy": 0.0}

    correct = (preds == labels) & mask
    return {"token_accuracy": correct.sum() / total}


def load_json_as_lists(path: Path):
    with path.open(encoding="utf‑8") as f:
        data = json.load(f)
    prompts = [row["prompt"] for row in data]
    codes   = [row["code"]   for row in data]
    return prompts, codes


def tokenize_examples(batch, tokenizer, max_len, prefix):
    inputs   = [prefix + p for p in batch["prompt"]]
    model_in = tokenizer(
        inputs, max_length=max_len, truncation=True, padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["code"], max_length=max_len, truncation=True, padding="max_length"
        )
    model_in["labels"] = labels["input_ids"]
    return model_in


# ─────────────────────────────────────────────────────────────
# 2.  main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Fine‑tune CodeT5 on CampEdUI")
    # data / paths
    parser.add_argument("--train_file", required=True, type=Path)
    parser.add_argument("--val_file",   required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    # model & training
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base")
    parser.add_argument("--num_train_epochs",   type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size",  type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate",  type=float, default=3e-5)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--warmup_steps",   type=int, default=200)
    parser.add_argument("--logging_steps",  type=int, default=50)
    parser.add_argument("--eval_steps",     type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # reproducibility & device info
    set_seed(args.seed)
    device = check_cuda()

    # 1) read JSON
    train_prompts, train_codes = load_json_as_lists(args.train_file)
    val_prompts,   val_codes   = load_json_as_lists(args.val_file)

    # 2) build datasets
    train_ds = Dataset.from_dict({"prompt": train_prompts, "code": train_codes})
    val_ds   = Dataset.from_dict({"prompt":   val_prompts, "code":   val_codes})

    # 3) tokenizer + model
    print(f"→ Loading '{args.model_name_or_path}' …")
    tok   = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # 4) tokenization
    prefix = "Generate CampEdUI component: "
    tokenized_train = train_ds.map(
        lambda b: tokenize_examples(b, tok, args.max_seq_length, prefix),
        batched=True,
        remove_columns=["prompt", "code"],
        load_from_cache_file=False,
    )
    tokenized_val = val_ds.map(
        lambda b: tokenize_examples(b, tok, args.max_seq_length, prefix),
        batched=True,
        remove_columns=["prompt", "code"],
        load_from_cache_file=False,
    )

    # 5) collator
    collator = DataCollatorForSeq2Seq(tok, model=model)

    # 6) training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        # --- remove evaluation_strategy, eval_steps, do_eval, etc. ---
        save_strategy="steps",
        save_steps=args.eval_steps,       # you can still checkpoint every X steps
        save_total_limit=2,

        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # only matters if you later run eval
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,

        num_train_epochs=args.num_train_epochs,
        logging_dir=str(Path(args.output_dir) / "logs"),
        logging_steps=args.logging_steps,
        fp16=args.fp16,
    )

    # 7) trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_token_accuracy,
    )

    # 8) train
    print("→ Starting fine‑tuning …")
    trainer.train()

    # 9) save
    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    print(f"🏁 Training complete — artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()