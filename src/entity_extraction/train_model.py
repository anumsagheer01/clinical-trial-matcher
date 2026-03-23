"""
Fine-tune Flan-T5-Small for medical entity extraction using PEFT (LoRA).

Task: Given patient text, extract structured medical entities as JSON.

PEFT/LoRA settings:
- r=16: Rank of adapter matrices. Higher = more capacity but slower.
- lora_alpha=32: Scaling factor. Usually 2*r.
- lora_dropout=0.1: Prevents overfitting.
- target_modules: Which layers get LoRA adapters.
"""

import json
import os
import time

import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import mlflow

MODEL_NAME = "google/flan-t5-small"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 256
TRAIN_DATA = os.path.join("data", "processed", "entity_training", "train.json")
VAL_DATA = os.path.join("data", "processed", "entity_training", "val.json")
OUTPUT_DIR = os.path.join("models", "entity_extractor")


class EntityExtractionDataset(Dataset):
    """Custom dataset for entity extraction."""

    def __init__(self, data_path, tokenizer, max_input_len=256, max_output_len=256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = f"Extract medical entities from this patient description: {example['text']}"
        target_text = json.dumps(example["entities"], ensure_ascii=False)

        input_encoding = self.tokenizer(
            input_text, max_length=self.max_input_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target_text, max_length=self.max_output_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def train():
    """Run the fine-tuning process."""
    print("=" * 60)
    print("FINE-TUNING ENTITY EXTRACTION MODEL")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n  Loading model: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Original model parameters: {total_params:,}")

    print("  Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    print(f"\n  Loading training data...")
    train_dataset = EntityExtractionDataset(TRAIN_DATA, tokenizer)
    val_dataset = EntityExtractionDataset(VAL_DATA, tokenizer)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n  Starting training...")
    start_time = time.time()

    mlflow.set_experiment("entity-extraction")
    with mlflow.start_run():
        mlflow.log_params({
            "model": MODEL_NAME,
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
        })


    trainer.train()

    mlflow.log_metrics({
                "train_loss": trainer.state.log_history[-1].get("train_loss", 0),
    })
    mlflow.log_artifact(os.path.join(OUTPUT_DIR, "lora_adapter"))

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed / 60:.1f} minutes")

    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  Saved LoRA adapter to {adapter_path}")

    adapter_size = sum(
        os.path.getsize(os.path.join(adapter_path, f))
        for f in os.listdir(adapter_path)
        if os.path.isfile(os.path.join(adapter_path, f))
    )
    print(f"  Adapter size: {adapter_size / (1024 * 1024):.1f} MB")

    return model, tokenizer


if __name__ == "__main__":
    train()