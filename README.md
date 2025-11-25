# Fake Job Posting Prediction

**A compact, end-to-end project to detect fake job postings using machine learning**

---

## Project overview

This repository contains code, artifacts and visualizations used to build a model that predicts whether a job posting is fraudulent. The project includes exploratory data analysis (EDA), preprocessing, feature engineering, model training (many candidate models), model selection, and a Streamlit-based UI to try predictions.

Key capabilities:

* Clean and transform raw job-posting text and metadata
* Extract numeric and text features (TF-IDF, SVD, one-hot, scaling)
* Train and evaluate multiple models and ensembles
* Persist preprocessing artifacts and trained models for production use
* Interactive demo via `app.py` (Streamlit)

---

## Quick start

1. Clone the repo.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Repository structure

```
Fake-Job-Posting-Prediction/
├─ artifacts/                # Persisted preprocessor artifacts and serialized arrays
│  ├─ X_processed.npz
│  ├─ feature_scaler.pkl
│  ├─ feature_selector.pkl
│  ├─ numeric_feature_names.pkl
│  ├─ ohe_columns.pkl
│  ├─ tfidf_vectorizer.pkl
│  └─ y_target.pkl
├─ assets/                   # Static assets (stylesheets)
│  └─ style.css
├─ data/                     # Datasets
│  └─ fake_job_postings.csv
├─ eda_plots/                # EDA images and visualizations
│  ├─ 01_wordcount_histograms.png
│  ├─ 02_wordcount_boxplot.png
│  └─ ...
├─ models/                   # Trained models and metrics
│  ├─ best_lgb_model.pkl
│  ├─ LogisticRegression_best.pkl
│  ├─ ensemble_models.pkl
│  ├─ xgbboost_optimized_model.pkl
│  └─ ensemble_metrics.json
├─ pages/                    # Streamlit multi-page scripts (or modular scripts)
│  ├─ 1_EDA_Full.py
│  ├─ 2_Preprocessing_Full.py
│  ├─ 3_Models.py
│  └─ 4_Predict.py
├─ preprocessing_logs/       # Logs produced during preprocessing runs
│  └─ preprocess_log_YYYY-MM-DD_hh-mm-ss.txt
├─ utils/                    # Utility modules (helpers, wrappers)
├─ .gitignore
├─ app.py                    # Streamlit entrypoint
├─ Fake_Job_Posting_Prediction.ipynb  # Notebook walkthrough (uploaded)
├─ requirements.txt
├─ scripts_eda.py            # EDA helper script
├─ scripts_preprocess.py     # Preprocessing pipeline script
├─ styles.css
└─ test.csv
```

---

## How to use the saved artifacts & models

Example: load preprocessing pipeline artifacts and run a prediction using a saved model.

```python
import joblib
import numpy as np

# load artifacts
tfidf = joblib.load('artifacts/tfidf_vectorizer.pkl')
scaler = joblib.load('artifacts/feature_scaler.pkl')
selector = joblib.load('artifacts/feature_selector.pkl')
model = joblib.load('models/best_lgb_model.pkl')

# prepare a single sample (example)
sample_text = "Senior Python developer needed..."
# run through tfidf or appropriate text pipeline, combine with numeric features, then:
# X_transformed = pipeline_transform(sample_text, numeric_features, tfidf, selector, scaler)
# y_pred = model.predict(X_transformed)
```

If you need the exact preprocessing order, check `scripts_preprocess.py` and the `pages/2_Preprocessing_Full.py` script for the canonical pipeline implementation.

---

## Notebooks and EDA

* `Fake_Job_Posting_Prediction.ipynb` — full exploratory notebook and modeling walkthrough.
* `eda_plots/` — ready-made PNGs used in the report/UI. Use them to quickly inspect distributions, correlations, and text analyses.

---

## Reproducing training

1. Ensure `data/fake_job_postings.csv` is present.
2. Run the preprocessing pipeline (`scripts_preprocess.py`) or `pages/2_Preprocessing_Full.py` to generate `artifacts/` content.
3. Run `pages/3_Models.py` (or the notebook) to train candidate models. Trained artifacts will be written to `models/`.
4. Evaluate results using the saved `*.json` metrics files and `eda_plots/` visual outputs.

---

## Deployment notes

* `app.py` is built with Streamlit and uses the artifacts stored in `/artifacts` and `/models` to make predictions.
* Keep the `artifacts` and `models` directories in the same relative location as `app.py`.
* For production: consider packaging the preprocessing pipeline as a single `sklearn.pipeline.Pipeline` object (so you only need to `joblib.load` one file).

---

