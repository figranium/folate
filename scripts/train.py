import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ELECTRA cross-encoder for Folate.")
    parser.add_argument("--model-name", default="google/electra-base-discriminator")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--save-dir", default="./folate-base")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(args.data_dir, "train.jsonl"),
            "validation": os.path.join(args.data_dir, "val.jsonl"),
        },
    )

    def preprocess(batch):
        tokenized = tokenizer(
            batch["query"],
            batch["element"],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = [float(x) for x in batch["label"]]
        return tokenized

    tokenized_ds = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["query", "element", "label"],
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir="./folate-checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved Folate base model to {args.save_dir}")


if __name__ == "__main__":
    main()
