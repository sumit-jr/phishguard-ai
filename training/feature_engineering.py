import pandas as pd
import re
import math

from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

# =========================================================
# PATHS
# =========================================================

DATASET_PATH = Path(
    "datasets/processed/final_phishing_dataset.csv"
)

TOP_DOMAINS_PATH = Path(
    "datasets/raw/top-1m.csv"
)

OUTPUT_PATH = Path(
    "datasets/processed/final_featured_dataset.csv"
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\n==============================")
print("LOADING DATASET")
print("==============================\n")

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded:")
print(df.shape)

# =========================================================
# LOAD TRUSTED DOMAINS
# =========================================================

print("\n==============================")
print("LOADING TOP DOMAINS")
print("==============================\n")

top_domains = pd.read_csv(
    TOP_DOMAINS_PATH,
    header=None
)

trusted_domains = set(
    top_domains[1]
    .astype(str)
    .str.lower()
)

print("Trusted Domains Loaded:")
print(len(trusted_domains))

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
    "ow.ly",
    "buff.ly"
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
# URL EXTRACTION
# =========================================================

def extract_urls(text):

    text = str(text)

    return re.findall(
        r'https?://\S+|www\.\S+',
        text
    )

# =========================================================
# DOMAIN EXTRACTION
# =========================================================

def get_domain(url):

    try:

        domain = urlparse(url).netloc

        return domain.lower()

    except:

        return ""

# =========================================================
# FEATURE: NUMBER OF URLS
# =========================================================

def num_urls(text):

    return len(
        extract_urls(text)
    )

# =========================================================
# FEATURE: IP URL
# =========================================================

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

# =========================================================
# FEATURE: SHORTENER
# =========================================================

def has_shortener(text):

    text = str(text).lower()

    for shortener in SHORTENERS:

        if shortener in text:
            return 1

    return 0

# =========================================================
# FEATURE: TRUSTED DOMAIN
# =========================================================

def has_trusted_domain(text):

    urls = extract_urls(text)

    for url in urls:

        domain = get_domain(url)

        if domain in trusted_domains:
            return 1

    return 0

# =========================================================
# FEATURE: SUSPICIOUS TLD
# =========================================================

def has_suspicious_tld(text):

    urls = extract_urls(text)

    for url in urls:

        domain = get_domain(url)

        for tld in SUSPICIOUS_TLDS:

            if domain.endswith(tld):
                return 1

    return 0

# =========================================================
# FEATURE: SUBDOMAIN COUNT
# =========================================================

def max_subdomain_count(text):

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

# =========================================================
# FEATURE: ENTROPY
# =========================================================

def calculate_entropy(text):

    text = str(text)

    if len(text) == 0:
        return 0

    counter = Counter(text)

    probabilities = [
        count / len(text)
        for count in counter.values()
    ]

    entropy = -sum(
        p * math.log2(p)
        for p in probabilities
    )

    return entropy

# =========================================================
# FEATURE: KEYWORD SCORE
# =========================================================

def keyword_score(text):

    text = str(text).lower()

    score = 0

    for keyword in PHISHING_KEYWORDS:

        if keyword in text:
            score += 1

    return score

# =========================================================
# FEATURE: EXCLAMATIONS
# =========================================================

def num_exclamations(text):

    return str(text).count("!")

# =========================================================
# FEATURE: DIGIT RATIO
# =========================================================

def digit_ratio(text):

    text = str(text)

    digits = sum(
        c.isdigit()
        for c in text
    )

    return digits / max(len(text), 1)

# =========================================================
# FEATURE: UPPERCASE RATIO
# =========================================================

def uppercase_ratio(text):

    text = str(text)

    uppercase = sum(
        c.isupper()
        for c in text
    )

    return uppercase / max(len(text), 1)

# =========================================================
# FEATURE: SPECIAL CHAR RATIO
# =========================================================

def special_char_ratio(text):

    text = str(text)

    special = sum(
        not c.isalnum() and not c.isspace()
        for c in text
    )

    return special / max(len(text), 1)

# =========================================================
# FEATURE: MESSAGE LENGTH
# =========================================================

def message_length(text):

    return len(str(text))

# =========================================================
# APPLY FEATURES
# =========================================================

print("\n==============================")
print("EXTRACTING FEATURES")
print("==============================\n")

df["num_urls"] = df["text"].apply(
    num_urls
)

df["has_ip_url"] = df["text"].apply(
    has_ip_url
)

df["has_shortener"] = df["text"].apply(
    has_shortener
)

df["trusted_domain"] = df["text"].apply(
    has_trusted_domain
)

df["has_suspicious_tld"] = df["text"].apply(
    has_suspicious_tld
)

df["subdomain_count"] = df["text"].apply(
    max_subdomain_count
)

df["entropy"] = df["text"].apply(
    calculate_entropy
)

df["keyword_score"] = df["text"].apply(
    keyword_score
)

df["num_exclamations"] = df["text"].apply(
    num_exclamations
)

df["digit_ratio"] = df["text"].apply(
    digit_ratio
)

df["uppercase_ratio"] = df["text"].apply(
    uppercase_ratio
)

df["special_char_ratio"] = df["text"].apply(
    special_char_ratio
)

df["message_length"] = df["text"].apply(
    message_length
)

# =========================================================
# SAVE FEATURED DATASET
# =========================================================

print("\n==============================")
print("SAVING FEATURED DATASET")
print("==============================\n")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Saved To:")
print(OUTPUT_PATH)

# =========================================================
# FINAL PREVIEW
# =========================================================

print("\n==============================")
print("FEATURED DATASET PREVIEW")
print("==============================\n")

print(df.head())

# =========================================================
# DATASET INFO
# =========================================================

print("\n==============================")
print("FEATURED DATASET INFO")
print("==============================\n")

print(df.info())