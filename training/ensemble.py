import re
import math
import joblib
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# =========================================================
# LOAD SVM MODEL
# =========================================================

print("Loading SVM model...")

svm_model = joblib.load(
    "models/svm/svm_model.pkl"
)

svm_vectorizer = joblib.load(
    "models/svm/tfidf_vectorizer.pkl"
)

svm_scaler = joblib.load(
    "models/svm/feature_scaler.pkl"
)

print("SVM loaded")

# =========================================================
# LOAD TRUSTED DOMAINS
# =========================================================

print("Loading trusted domains...")

top_domains = pd.read_csv(
    "datasets/raw/top-1m.csv",
    header=None
)

trusted_domains = set(
    top_domains[1]
    .astype(str)
    .str.lower()
)

print("Trusted domains loaded")

# =========================================================
# CONFIG
# =========================================================

SUSPICIOUS_TLDS = [
    ".tk",
    ".ru",
    ".xyz",
    ".top",
    ".gq",
    ".ml"
]

SHORTENERS = [
    "bit.ly",
    "tinyurl",
    "goo.gl",
    "t.co",
    "ow.ly"
]

PHISHING_KEYWORDS = [
    "verify",
    "urgent",
    "account",
    "suspended",
    "click",
    "password",
    "bank",
    "security",
    "confirm",
    "payment",
    "invoice",
    "gift",
    "crypto",
    "wallet",
    "login",
    "signin",
    "update",
    "limited",
    "alert"
]

# =========================================================
# TRUSTED BRANDS
# =========================================================

TRUSTED_BRANDS = [
    "github",
    "google",
    "microsoft",
    "amazon",
    "slack",
    "dropbox",
    "notion",
    "atlassian",
    "paypal"
]

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

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\.\S+",
        " URLTOKEN ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " EMAILTOKEN ",
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9@._:/\-\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =========================================================
# URL EXTRACTION
# =========================================================

def extract_urls(text):

    return re.findall(
        r'https?://\S+|www\.\S+',
        text
    )

# =========================================================
# DOMAIN EXTRACTION
# =========================================================

def get_domain(url):

    try:
        return urlparse(url).netloc.lower()

    except:
        return ""

# =========================================================
# FEATURE FUNCTIONS
# =========================================================

def num_urls(text):

    return len(
        extract_urls(text)
    )

def has_ip_url(text):

    pattern = (
        r'http[s]?://'
        r'(?:\d{1,3}\.){3}\d{1,3}'
    )

    return int(
        bool(
            re.search(pattern, text)
        )
    )

def has_shortener(text):

    text = text.lower()

    for shortener in SHORTENERS:

        if shortener in text:
            return 1

    return 0

def trusted_domain(text):

    urls = extract_urls(text)

    for url in urls:

        domain = get_domain(url)

        if domain in trusted_domains:
            return 1

    return 0

def suspicious_tld(text):

    urls = extract_urls(text)

    for url in urls:

        domain = get_domain(url)

        for tld in SUSPICIOUS_TLDS:

            if domain.endswith(tld):
                return 1

    return 0

def subdomain_count(text):

    urls = extract_urls(text)

    max_count = 0

    for url in urls:

        domain = get_domain(url)

        count = domain.count(".")

        max_count = max(
            max_count,
            count
        )

    return max_count

def entropy(text):

    text = str(text)

    if len(text) == 0:
        return 0

    counter = Counter(text)

    probabilities = [
        count / len(text)
        for count in counter.values()
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
    )

def keyword_score(text):

    text = text.lower()

    score = 0

    for keyword in PHISHING_KEYWORDS:

        if keyword in text:
            score += 1

    return score

# =========================================================
# TRUSTED BRAND DETECTION
# =========================================================

def trusted_brand_detected(text):

    text = text.lower()

    for brand in TRUSTED_BRANDS:

        if brand in text:
            return True

    return False

def num_exclamations(text):

    return text.count("!")

def digit_ratio(text):

    digits = sum(
        c.isdigit()
        for c in text
    )

    return digits / max(
        len(text),
        1
    )

def uppercase_ratio(text):

    uppercase = sum(
        c.isupper()
        for c in text
    )

    return uppercase / max(
        len(text),
        1
    )

def special_char_ratio(text):

    special = sum(
        not c.isalnum() and not c.isspace()
        for c in text
    )

    return special / max(
        len(text),
        1
    )

def message_length(text):

    return len(text)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(text):

    return [[
        num_urls(text),
        has_ip_url(text),
        has_shortener(text),
        trusted_domain(text),
        suspicious_tld(text),
        subdomain_count(text),
        entropy(text),
        keyword_score(text),
        num_exclamations(text),
        digit_ratio(text),
        uppercase_ratio(text),
        special_char_ratio(text),
        message_length(text)
    ]]

# =========================================================
# SVM PREDICTION
# =========================================================

def svm_predict(text):

    cleaned = clean_text(text)

    text_vector = svm_vectorizer.transform(
        [cleaned]
    )

    features = extract_features(cleaned)

    features_df = pd.DataFrame(
        features,
        columns=FEATURE_COLUMNS
    )

    scaled = svm_scaler.transform(
        features_df
    )

    final_input = hstack([
        text_vector,
        csr_matrix(scaled)
    ])

    score = svm_model.decision_function(
        final_input
    )[0]

    probability = 1 / (
        1 + np.exp(-score)
    )

    return probability

# =========================================================
# LAZY LOADED BERT
# =========================================================

def bert_predict(text):

    print("\nLoading BERT model...")

    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification
    )

    BERT_PATH = Path(
        "models/bert"
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        BERT_PATH
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_PATH
    )

    model.eval()

    print("BERT loaded")

    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    return probabilities[0][1].item()

# =========================================================
# THREAT REASONS
# =========================================================

def threat_reasons(text):

    reasons = []

    lowered = text.lower()

    if keyword_score(lowered) >= 2:
        reasons.append(
            "Multiple phishing keywords detected"
        )

    if has_shortener(lowered):
        reasons.append(
            "Shortened URL detected"
        )

    if suspicious_tld(lowered):
        reasons.append(
            "Suspicious domain TLD detected"
        )

    if num_urls(lowered) > 2:
        reasons.append(
            "Multiple URLs detected"
        )

    if entropy(lowered) > 4.5:
        reasons.append(
            "High entropy suspicious text"
        )

    if has_ip_url(lowered):
        reasons.append(
            "IP-based URL detected"
        )

    return reasons

# =========================================================
# CASCADE DETECTOR
# =========================================================

def detect_phishing(text):

    # =====================================================
    # PRIMARY FAST DETECTOR
    # =====================================================

    svm_score = svm_predict(text)

    # =====================================================
    # BERT DISABLED FOR CLOUD DEPLOYMENT
    # =====================================================

    bert_score = 0

    bert_used = False

    # =====================================================
    # FINAL ENSEMBLE SCORE
    # =====================================================

    final_score = (
        0.6 * svm_score +
        0.4 * bert_score
    )

    # =====================================================
    # TRUSTED BRAND ADJUSTMENT
    # =====================================================

    if trusted_brand_detected(text):

        if not suspicious_tld(text):

            final_score *= 0.75

    # =====================================================
    # FINAL LABEL
    # =====================================================

    if final_score >= 0.5:

        label = "PHISHING"

    else:

        label = "LEGITIMATE"

    confidence = round(
        final_score * 100,
        2
    )

    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {
        "label": label,
        "confidence": confidence,
        "svm_score": round(svm_score, 4),
        "bert_score": round(bert_score, 4),
        "bert_used": bert_used,
        "threat_reasons": threat_reasons(text)
    }

# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":

    print("\n==============================")
    print("CASCADE PHISHING DETECTOR")
    print("==============================")

    while True:

        print("\nEnter Email Text:\n")

        email = input()

        result = detect_phishing(
            email
        )

        print("\n==============================")
        print("RESULT")
        print("==============================\n")

        print(
            "Prediction:",
            result["label"]
        )

        print(
            "Confidence:",
            f"{result['confidence']}%"
        )

        print("\nMODEL SCORES\n")

        print(
            "SVM:",
            result["svm_score"]
        )

        print(
            "BERT:",
            result["bert_score"]
        )

        print(
            "\nBERT Used:",
            result["bert_used"]
        )

        print("\nTHREAT REASONS\n")

        if len(result["threat_reasons"]) == 0:

            print("No major threats detected")

        else:

            for reason in result["threat_reasons"]:

                print("-", reason)

        print("\n==============================")