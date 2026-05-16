import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from xgboost import XGBClassifier

# =========================================================
# PATHS
# =========================================================

DATASET_PATH = Path(
    "datasets/processed/final_featured_dataset.csv"
)

MODEL_DIR = Path(
    "models/xgboost"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"

# =========================================================
# LOAD DATASET
# =========================================================

print("\n==============================")
print("LOADING DATASET")
print("==============================\n")

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:")
print(df.shape)

# =========================================================
# FEATURES
# =========================================================

TEXT_COLUMN = "text"

FEATURE_COLUMNS = [
    "num_urls",
    "has_ip_url",
    "has_shortener",
    "trusted_domain",
    "has_suspicious_tld",
    "subdomain_count",
    "entropy",
    "keyword_score",
    "num_exclamations",
    "digit_ratio",
    "uppercase_ratio",
    "special_char_ratio",
    "message_length"
]

TARGET_COLUMN = "label"

# =========================================================
# INPUTS
# =========================================================

X_text = df[TEXT_COLUMN]

X_features = df[FEATURE_COLUMNS]

y = df[TARGET_COLUMN]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================\n")

X_train_text, X_test_text, \
X_train_features, X_test_features, \
y_train, y_test = train_test_split(
    X_text,
    X_features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train_text))
print("Testing Samples:", len(X_test_text))

# =========================================================
# TF-IDF
# =========================================================

print("\n==============================")
print("TF-IDF VECTORIZATION")
print("==============================\n")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(
    X_train_text
)

X_test_tfidf = vectorizer.transform(
    X_test_text
)

print("TF-IDF Shape:")
print(X_train_tfidf.shape)

# =========================================================
# SCALE FEATURES
# =========================================================

print("\n==============================")
print("SCALING FEATURES")
print("==============================\n")

scaler = StandardScaler()

X_train_features_scaled = scaler.fit_transform(
    X_train_features
)

X_test_features_scaled = scaler.transform(
    X_test_features
)

X_train_features_sparse = csr_matrix(
    X_train_features_scaled
)

X_test_features_sparse = csr_matrix(
    X_test_features_scaled
)

# =========================================================
# COMBINE FEATURES
# =========================================================

print("\n==============================")
print("COMBINING FEATURES")
print("==============================\n")

X_train_final = hstack([
    X_train_tfidf,
    X_train_features_sparse
])

X_test_final = hstack([
    X_test_tfidf,
    X_test_features_sparse
])

print("Final Shape:")
print(X_train_final.shape)

# =========================================================
# TRAIN XGBOOST
# =========================================================

print("\n==============================")
print("TRAINING XGBOOST")
print("==============================\n")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    tree_method="hist"
)

model.fit(
    X_train_final,
    y_train
)

print("Training Complete")

# =========================================================
# PREDICTIONS
# =========================================================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================\n")

y_pred = model.predict(
    X_test_final
)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
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
        y_test,
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
    y_test,
    y_pred
)

print(cm)

# =========================================================
# SAVE MODEL
# =========================================================

print("\n==============================")
print("SAVING MODEL")
print("==============================\n")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print("Model Saved:")
print(MODEL_PATH)

print("\nVectorizer Saved:")
print(VECTORIZER_PATH)

print("\nScaler Saved:")
print(SCALER_PATH)

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================\n")