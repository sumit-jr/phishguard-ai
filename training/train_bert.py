import pandas as pd
import numpy as np
import torch

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================================================
# PATHS
# =========================================================

DATASET_PATH = Path(
    "datasets/processed/final_phishing_dataset.csv"
)

MODEL_DIR = Path(
    "models/bert"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\n==============================")
print("LOADING DATASET")
print("==============================\n")

df = pd.read_csv(DATASET_PATH)

print("Original Dataset Shape:")
print(df.shape)

# =========================================================
# KEEP REQUIRED COLUMNS
# =========================================================

df = df[["text", "label"]]

df["label"] = df["label"].astype(int)

# =========================================================
# FAST TRAINING MODE
# =========================================================

print("\n==============================")
print("FAST TRAINING MODE")
print("==============================\n")

# ---------------------------------------------------------
# RECOMMENDED FOR LOCAL TRAINING
# ---------------------------------------------------------

# df = df.sample(
#     n=50000,
#     random_state=42
# )

# ---------------------------------------------------------
# FULL DATASET TRAINING (UNCOMMENT LATER)
# ---------------------------------------------------------

df = df.sample(
    n=200000,
    random_state=42
)

print("Training Dataset Shape:")
print(df.shape)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================\n")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Training Samples:", len(train_df))
print("Testing Samples:", len(test_df))

# =========================================================
# TOKENIZER
# =========================================================

print("\n==============================")
print("LOADING TOKENIZER")
print("==============================\n")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# =========================================================
# DATASETS
# =========================================================

print("\n==============================")
print("TOKENIZING DATASET")
print("==============================\n")

train_dataset = Dataset.from_pandas(
    train_df
)

test_dataset = Dataset.from_pandas(
    test_df
)

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# remove unnecessary columns
train_dataset = train_dataset.remove_columns(
    ["text"]
)

test_dataset = test_dataset.remove_columns(
    ["text"]
)

# rename labels
train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)

# pytorch format
train_dataset.set_format("torch")

test_dataset.set_format("torch")

# =========================================================
# LOAD MODEL
# =========================================================

print("\n==============================")
print("LOADING DISTILBERT")
print("==============================\n")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions
    )

    recall = recall_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="epoch",

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    learning_rate=2e-5,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=False,

    report_to="none"
)

# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\n==============================")
print("TRAINING DISTILBERT")
print("==============================\n")

trainer.train()

# =========================================================
# FINAL EVALUATION
# =========================================================

print("\n==============================")
print("FINAL EVALUATION")
print("==============================\n")

predictions = trainer.predict(
    test_dataset
)

y_pred = np.argmax(
    predictions.predictions,
    axis=1
)

y_true = predictions.label_ids

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

# =========================================================
# SAVE MODEL
# =========================================================

print("\n==============================")
print("SAVING MODEL")
print("==============================\n")

trainer.save_model(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)

print("Model Saved To:")
print(MODEL_DIR)

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================\n")