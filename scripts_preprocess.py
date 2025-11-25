# ===============================================
#  OPTIMIZED PREPROCESSING SCRIPT (READY FOR STREAMLIT)
#  Saves artifacts + logs into dedicated folders
# ===============================================

import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import sparse
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import datetime

# -------------------------------
# 0. SETUP FOLDERS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fake_job_postings.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
LOG_DIR = os.path.join(BASE_DIR, "preprocessing_logs")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"preprocess_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

# Redirect console output to log file
logger = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = logger

# -------------------------------
# NLTK Downloads
# -------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("=" * 70)
print("OPTIMIZED PREPROCESSING FOR IMBALANCED FAKE JOB DETECTION")
print("=" * 70)

# -------------------------------
# 1. Load Dataset
# -------------------------------
print("\n[1] Loading dataset...")

if not os.path.exists(DATA_PATH):
    print("❌ ERROR: Dataset file 'fake_job_postings.csv' not found.")
    print("Place the file in the project root folder.")
    sys.exit()

df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
print(f"✅ Dataset loaded: {df.shape}")
print(f"✅ Fraud ratio: {df['fraudulent'].mean()*100:.2f}%")

# -------------------------------
# 2. Column Groups
# -------------------------------
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
cat_cols = ['location', 'department', 'employment_type', 'required_experience',
            'required_education', 'industry', 'function']
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']

# -------------------------------
# 3. Missing Values
# -------------------------------
print("\n[2] Handling Missing Values")
print("-" * 60)

# Text columns
for col in text_cols:
    df[col] = df[col].fillna('')
    print(f"Filled missing in text column: {col}")

# Categorical columns
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')
    print(f"Filled missing in categorical column: {col}")

# Binary columns
for col in binary_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    print(f"Filled missing in binary column: {col}")

# Salary column → convert to binary indicator
df['salary_specified'] = df['salary_range'].apply(
    lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
)
df.drop(columns=['salary_range'], inplace=True)
print("Created binary salary_specified")

# -------------------------------
# 4. Reduce Cardinality
# -------------------------------
print("\n[3] Reducing Categorical Cardinality")
print("-" * 60)

def reduce_cardinality(series, threshold=50, other_label='Other'):
    value_counts = series.value_counts()
    keep = value_counts[value_counts >= threshold].index
    reduced = series.where(series.isin(keep), other_label)
    print(f"{series.name}: {series.nunique()} → {reduced.nunique()} categories")
    return reduced

df['location'] = reduce_cardinality(df['location'], 100)
df['department'] = reduce_cardinality(df['department'], 50)
df['industry'] = reduce_cardinality(df['industry'], 30)

# -------------------------------
# 5. Text Cleaning
# -------------------------------
print("\n[4] Cleaning Text")
print("-" * 60)

df['text_raw'] = df[text_cols].agg(' '.join, axis=1)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df['text_clean'] = df['text_raw'].apply(clean_text)
df['text_clean'] = df['text_clean'].replace("", "emptytext")

print("✅ Text cleaned successfully")

# -------------------------------
# 6. Numeric Text Features
# -------------------------------
print("\n[5] Extracting Numeric Text Features")
print("-" * 60)

def extract_text_features(row):
    raw = row['text_raw']
    clean = row['text_clean']
    words = clean.split()
    wc = len(words)

    up_words = sum(1 for w in raw.split() if w.isupper() and len(w) > 1)

    return pd.Series({
        'char_count': len(raw),
        'word_count': wc,
        'unique_words': len(set(words)),
        'avg_word_len': (sum(len(w) for w in words) / wc) if wc else 0,
        'num_exclaims': raw.count('!'),
        'num_questions': raw.count('?'),
        'has_email': 1 if '@' in raw else 0,
        'has_url': 1 if 'http' in raw or 'www' in raw else 0,
        'all_caps_ratio': sum(c.isupper() for c in raw) / len(raw) if len(raw) else 0,
        'uppercase_word_count': up_words,
        'text_richness': len(set(words)) / wc if wc else 0
    })

numeric_cols = [
    'char_count', 'word_count', 'unique_words', 'avg_word_len',
    'num_exclaims', 'num_questions', 'has_email', 'has_url',
    'all_caps_ratio', 'uppercase_word_count', 'text_richness'
]

df[numeric_cols] = df.apply(extract_text_features, axis=1)
print("✅ Numeric features extracted")

# -------------------------------
# 7. One-Hot Encoding
# -------------------------------
print("\n[6] Encoding Categorical Variables")
print("-" * 60)

df_cat = pd.get_dummies(df[cat_cols], drop_first=True)
print(f"✅ Encoded categorical → {df_cat.shape[1]} features")

# -------------------------------
# 8. TF-IDF Vectorization
# -------------------------------
print("\n[7] TF-IDF Vectorization")
print("-" * 60)

tfidf = TfidfVectorizer(
    max_features=2000,
    min_df=10,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

X_text = tfidf.fit_transform(df['text_clean'])
print("✅ TF-IDF completed:", X_text.shape)

# -------------------------------
# 9. Scaling Numeric Features
# -------------------------------
print("\n[8] Scaling Numeric Features")

continuous_cols = [
    'char_count', 'word_count', 'unique_words', 'avg_word_len',
    'all_caps_ratio', 'uppercase_word_count', 'text_richness'
]
binary_extra = ['num_exclaims', 'num_questions', 'has_email', 'has_url'] + binary_cols + ['salary_specified']

scaler = StandardScaler()
X_cont = scaler.fit_transform(df[continuous_cols])
X_bin = df[binary_extra].values

X_num = np.hstack([X_cont, X_bin])

# -------------------------------
# 10. Combine Features
# -------------------------------
print("\n[9] Combining Features")

X_combined = sparse.hstack([
    X_text,
    sparse.csr_matrix(X_num),
    sparse.csr_matrix(df_cat.values)
], format="csr")

print("✅ Combined feature matrix:", X_combined.shape)

# -------------------------------
# 11. Feature Selection
# -------------------------------
print("\n[10] Feature Selection (ANOVA F-test)")

fraud_count = df['fraudulent'].sum()
k_features = max(100, min(500, fraud_count * 2))

selector = SelectKBest(f_classif, k=k_features)
X_final = selector.fit_transform(X_combined, df['fraudulent'])

print("✅ Final features:", X_final.shape)

# -------------------------------
# 12. Save Artifacts
# -------------------------------
print("\n[11] Saving Artifacts")
# ✅ save one-hot encoded column list
pickle.dump(df_cat.columns.tolist(), open(os.path.join(ARTIFACT_DIR, "ohe_columns.pkl"), "wb"))

# ✅ save numeric feature order
pickle.dump(numeric_cols, open(os.path.join(ARTIFACT_DIR, "numeric_feature_names.pkl"), "wb"))

sparse.save_npz(os.path.join(ARTIFACT_DIR, "X_processed.npz"), X_final)
pickle.dump(df['fraudulent'], open(os.path.join(ARTIFACT_DIR, "y_target.pkl"), "wb"))
pickle.dump(tfidf, open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(ARTIFACT_DIR, "feature_scaler.pkl"), "wb"))
pickle.dump(selector, open(os.path.join(ARTIFACT_DIR, "feature_selector.pkl"), "wb"))

df.to_csv(os.path.join(ARTIFACT_DIR, "processed_job_postings.csv"), index=False)

print("\n✅ ALL ARTIFACTS SAVED in /artifacts/")
print(f"✅ LOG saved at: {LOG_FILE}")
print("\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY 🎉")

# Close the log file
logger.close()
