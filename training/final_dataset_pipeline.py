import pandas as pd
import re
from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_PATH = Path("datasets/raw")

OUTPUT_PATH = Path("datasets/processed")

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# TEXT CLEANING FUNCTION
# =========================================================

def clean_text(text):
    """
    Clean and normalize email text
    while preserving cybersecurity signals
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # lowercase
    text = text.lower()

    # replace urls with token
    text = re.sub(
        r"http\S+|www\.\S+",
        " URLTOKEN ",
        text
    )

    # replace email addresses with token
    text = re.sub(
        r"\S+@\S+",
        " EMAILTOKEN ",
        text
    )

    # remove html tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # preserve cybersecurity characters
    text = re.sub(
        r"[^a-zA-Z0-9@._:/\-\s]",
        " ",
        text
    )

    # remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =========================================================
# COMBINE MULTIPLE COLUMNS
# =========================================================

def combine_columns(df, columns):

    available_columns = [
        col for col in columns
        if col in df.columns
    ]

    if len(available_columns) == 0:
        return pd.Series([""] * len(df))

    return (
        df[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

# =========================================================
# STANDARDIZE DATASET FORMAT
# =========================================================

def create_dataset(text_series, labels):

    temp = pd.DataFrame()

    temp["text"] = text_series.astype(str)

    temp["label"] = labels

    return temp

# =========================================================
# DATASET CONTAINER
# =========================================================

all_datasets = []

print("\n==============================")
print("LOADING DATASETS")
print("==============================\n")

# =========================================================
# 1. fraud_email.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "fraud_email.csv"
    )

    temp = create_dataset(
        df["Text"],
        df["Class"]
    )

    all_datasets.append(temp)

    print("Loaded fraud_email.csv")

except Exception as e:

    print(
        "Error loading fraud_email.csv:",
        e
    )

# =========================================================
# 2. email_dataset_100k.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "email_dataset_100k.csv"
    )

    temp = create_dataset(
        df["raw_text"],
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded email_dataset_100k.csv")

except Exception as e:

    print(
        "Error loading email_dataset_100k.csv:",
        e
    )

# =========================================================
# 3. CEAS_08.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "CEAS_08.csv"
    )

    text_data = combine_columns(
        df,
        ["sender", "subject", "body"]
    )

    temp = create_dataset(
        text_data,
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded CEAS_08.csv")

except Exception as e:

    print(
        "Error loading CEAS_08.csv:",
        e
    )

# =========================================================
# 4. Enron.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "Enron.csv"
    )

    text_data = combine_columns(
        df,
        ["subject", "body", "message"]
    )

    temp = create_dataset(
        text_data,
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded Enron.csv")

except Exception as e:

    print(
        "Error loading Enron.csv:",
        e
    )

# =========================================================
# 5. Ling.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "Ling.csv"
    )

    text_data = combine_columns(
        df,
        ["subject", "body", "message"]
    )

    temp = create_dataset(
        text_data,
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded Ling.csv")

except Exception as e:

    print(
        "Error loading Ling.csv:",
        e
    )

# =========================================================
# 6. Nazario.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "Nazario.csv"
    )

    text_data = combine_columns(
        df,
        ["sender", "subject", "body"]
    )

    temp = create_dataset(
        text_data,
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded Nazario.csv")

except Exception as e:

    print(
        "Error loading Nazario.csv:",
        e
    )

# =========================================================
# 7. Nigerian_Fraud.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "Nigerian_Fraud.csv"
    )

    text_data = combine_columns(
        df,
        ["sender", "subject", "body"]
    )

    temp = create_dataset(
        text_data,
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded Nigerian_Fraud.csv")

except Exception as e:

    print(
        "Error loading Nigerian_Fraud.csv:",
        e
    )

# =========================================================
# 8. phishing_email.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "phishing_email.csv"
    )

    temp = create_dataset(
        df["text_combined"],
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded phishing_email.csv")

except Exception as e:

    print(
        "Error loading phishing_email.csv:",
        e
    )

# =========================================================
# 9. SpamAssasin.csv
# =========================================================

try:

    df = pd.read_csv(
        BASE_PATH / "SpamAssasin.csv"
    )

    text_data = combine_columns(
        df,
        ["sender", "subject", "body"]
    )

    temp = create_dataset(
        text_data,
        df["label"]
    )

    all_datasets.append(temp)

    print("Loaded SpamAssasin.csv")

except Exception as e:

    print(
        "Error loading SpamAssasin.csv:",
        e
    )

# =========================================================
# MERGE DATASETS
# =========================================================

print("\n==============================")
print("MERGING DATASETS")
print("==============================\n")

final_df = pd.concat(
    all_datasets,
    ignore_index=True
)

print(
    "Total Samples Before Cleaning:",
    len(final_df)
)

# =========================================================
# CLEAN DATASET
# =========================================================

print("\n==============================")
print("CLEANING DATASET")
print("==============================\n")

# remove null values
final_df.dropna(inplace=True)

# convert labels
final_df["label"] = pd.to_numeric(
    final_df["label"],
    errors="coerce"
)

# keep valid labels only
final_df = final_df[
    final_df["label"].isin([0, 1])
]

# convert text
final_df["text"] = final_df["text"].astype(str)

# clean text
final_df["text"] = final_df["text"].apply(
    clean_text
)

# remove short text
final_df = final_df[
    final_df["text"].str.len() > 25
]

# remove duplicates
final_df.drop_duplicates(
    subset=["text"],
    inplace=True
)

# remove empty rows
final_df = final_df[
    final_df["text"].str.strip() != ""
]

print(
    "Total Samples After Cleaning:",
    len(final_df)
)

# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\n==============================")
print("LABEL DISTRIBUTION")
print("==============================\n")

print(
    final_df["label"].value_counts()
)

# =========================================================
# BALANCE DATASET
# =========================================================

print("\n==============================")
print("BALANCING DATASET")
print("==============================\n")

# separate classes
phishing_df = final_df[
    final_df["label"] == 1
]

legit_df = final_df[
    final_df["label"] == 0
]

# minimum class size
min_class_size = min(
    len(phishing_df),
    len(legit_df)
)

# balance phishing
phishing_sample = phishing_df.sample(
    n=min_class_size,
    random_state=42
)

# balance legitimate
legit_sample = legit_df.sample(
    n=min_class_size,
    random_state=42
)

# combine
balanced_df = pd.concat(
    [phishing_sample, legit_sample],
    ignore_index=True
)

# shuffle
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(
    "Balanced Dataset Size:",
    len(balanced_df)
)

print("\nBalanced Label Distribution:\n")

print(
    balanced_df["label"].value_counts()
)

# =========================================================
# SAVE FINAL DATASET
# =========================================================

final_dataset_path = (
    OUTPUT_PATH /
    "final_phishing_dataset.csv"
)

balanced_df.to_csv(
    final_dataset_path,
    index=False
)

print("\n==============================")
print("FINAL DATASET SAVED")
print("==============================\n")

print("Saved To:")

print(final_dataset_path)

# =========================================================
# PREVIEW FINAL DATASET
# =========================================================

print("\n==============================")
print("FINAL DATASET PREVIEW")
print("==============================\n")

print(
    balanced_df.head()
)

# =========================================================
# FINAL DATASET INFO
# =========================================================

print("\n==============================")
print("FINAL DATASET INFO")
print("==============================\n")

print(
    balanced_df.info()
)