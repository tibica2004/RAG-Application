from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import classification_report
from setfit import (
    SetFitModel,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).parent

# Load dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": str(BASE_DIR / "train.jsonl"),
        "test": str(BASE_DIR / "test.jsonl"),
    },
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load base model
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Training arguments
args = TrainingArguments(
    batch_size=16,
    num_epochs=5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("=" * 60)
print("Training router...")
print("=" * 60)

trainer.train()

print("\nTraining finished!")

print("\nEvaluating...")

predictions = trainer.model.predict(test_dataset["text"])

print(
    classification_report(
        test_dataset["label"],
        predictions,
        digits=4
    )
)

# Save model
save_path = BASE_DIR.parent.parent / "models" / "router"

trainer.model.save_pretrained(save_path)

print("\n" + "=" * 60)
print(f"Router saved to:\n{save_path}")
print("=" * 60)