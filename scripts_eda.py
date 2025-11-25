# =======================================
# FULL EDA SCRIPT (WITH IMAGE SAVING)
# =======================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

# -------------------------------
# SETUP
# -------------------------------
os.makedirs("eda_plots", exist_ok=True)

def savefig(name):
    """Save figure automatically into eda_plots folder."""
    path = os.path.join("eda_plots", name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

print("✅ EDA started...")

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("fake_job_postings.csv", encoding="utf-8", on_bad_lines="skip")
print("✅ Dataset loaded:", df.shape)

# -------------------------------
# 2. TEXT STATS
# -------------------------------
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]

for col in text_cols:
    df[f"{col}_wordcount"] = df[col].fillna("").apply(lambda x: len(str(x).split()))
    df[f"{col}_charcount"] = df[col].fillna("").apply(lambda x: len(str(x)))

# ----------------------------------------------------------
# HISTOGRAMS — Word Count Distribution by Fraud Status
# ----------------------------------------------------------
plt.figure(figsize=(15,12))
for i, col in enumerate(text_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=df, x=f"{col}_wordcount", hue="fraudulent", bins=50, alpha=0.6)
    plt.title(f"Distribution of {col.capitalize()} Word Count by Fraud Status")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")

plt.tight_layout()
savefig("01_wordcount_histograms.png")

# ----------------------------------------------------------
# Boxplot for all text word counts
# ----------------------------------------------------------
plt.figure(figsize=(10,6))
long_form = df.melt(
    id_vars="fraudulent",
    value_vars=[f"{col}_wordcount" for col in text_cols],
    var_name="TextField",
    value_name="WordCount"
)
sns.boxplot(data=long_form, x="TextField", y="WordCount", hue="fraudulent")
plt.title("Word Count Distribution by Fraud Status (All Text Fields)")
plt.xticks(rotation=45)
plt.ylim(0, 1000)
savefig("02_wordcount_boxplot.png")

# -------------------------------
# 3. WORD CLOUDS & TOP WORDS
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join([w for w in text.split() if w not in stop_words])

fraud_text = " ".join(df[df["fraudulent"]==1]["description"].dropna().apply(clean_text))
real_text  = " ".join(df[df["fraudulent"]==0]["description"].dropna().apply(clean_text))

fraud_wc = WordCloud(width=800, height=400, background_color="black").generate(fraud_text)
real_wc  = WordCloud(width=800, height=400, background_color="white").generate(real_text)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(fraud_wc, interpolation="bilinear")
plt.axis("off"); plt.title("Fraudulent Postings WordCloud")

plt.subplot(1,2,2)
plt.imshow(real_wc, interpolation="bilinear")
plt.axis("off"); plt.title("Genuine Postings WordCloud")

savefig("03_wordclouds.png")

# -------------------------------
# Top 20 frequent words
# -------------------------------
def top_words(text, n=20):
    words = text.split()
    return Counter(words).most_common(n)

print("\nTop 20 Fraudulent Words:\n", top_words(fraud_text, 20))
print("\nTop 20 Genuine Words:\n", top_words(real_text, 20))

# -------------------------------
# 4. CORRELATION HEATMAP
# -------------------------------
num_cols = ["telecommuting", "has_company_logo", "has_questions", "fraudulent"]
corr = df[num_cols].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
savefig("04_correlation_heatmap.png")

# -------------------------------
# 5. FEATURE RELATIONSHIPS
# -------------------------------

# Employment type
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="employment_type", y="fraudulent", estimator=np.mean)
plt.title("Fraud Rate by Employment Type")
plt.xticks(rotation=45)
savefig("05_fraud_by_employment_type.png")

# Experience
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="required_experience", y="fraudulent", estimator=np.mean)
plt.title("Fraud Rate by Required Experience")
plt.xticks(rotation=45)
savefig("06_fraud_by_required_experience.png")

# Education
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="required_education", y="fraudulent", estimator=np.mean)
plt.title("Fraud Rate by Required Education")
plt.xticks(rotation=45)
savefig("07_fraud_by_required_education.png")

# Industry
top_industries = df["industry"].value_counts().head(10).index
plt.figure(figsize=(10,6))
sns.barplot(
    data=df[df["industry"].isin(top_industries)],
    x="industry", y="fraudulent", estimator=np.mean
)
plt.title("Fraud Rate by Industry (Top 10)")
plt.xticks(rotation=75)
savefig("08_fraud_by_industry_top10.png")

# -------------------------------
# 6. CROSS FEATURE HEATMAPS
# -------------------------------
categorical_cols = ['employment_type', 'has_company_logo', 'telecommuting', 'required_experience']

for i, col1 in enumerate(categorical_cols):
    for col2 in categorical_cols[i+1:]:
        pivot = pd.pivot_table(
            df, values="fraudulent",
            index=col1, columns=col2,
            aggfunc=np.mean
        )

        plt.figure(figsize=(8,5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title(f"Fraud Rate: {col1} vs {col2}")
        plt.ylabel(col1)
        plt.xlabel(col2)

        savefig(f"09_cross_{col1}_vs_{col2}.png")

# -------------------------------
# 7. BINARY COLUMN DISTRIBUTIONS
# -------------------------------
binary_cols = ["telecommuting", "has_company_logo", "has_questions"]

for col in binary_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, hue="fraudulent")
    plt.title(f"Distribution of {col} by Fraudulent Status")
    plt.xlabel(f"{col} (0=No, 1=Yes)")
    savefig(f"10_binary_dist_{col}.png")

# -------------------------------
# 8. Fraud Summary Tables (printed to terminal)
# -------------------------------
cat_cols = ["employment_type", "required_experience", "required_education", "industry"]
for col in cat_cols:
    summary = df.groupby(col)["fraudulent"].mean().sort_values(ascending=False)
    print(f"\nFraud Rate by {col}:")
    print(summary.head(10))

print("\n✅ EDA completed. All plots saved to eda_plots/")
