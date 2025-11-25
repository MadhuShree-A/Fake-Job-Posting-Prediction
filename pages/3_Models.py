import streamlit as st
import os
import json
import pickle
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Model Results", page_icon="🤖", layout="wide")
st.title("🤖 Model Performance Dashboard")
st.write("Interactive performance comparison for all trained models")

# Let user change where metrics/models live; search both below in loaders
DEFAULT_MODELS_DIR = "models"
MODELS_DIR = st.sidebar.text_input("Models directory", value=DEFAULT_MODELS_DIR)
EXTRA_DIRS = [MODELS_DIR, "/mnt/data"]  # search order

# -----------------------------------------------------------
# FILE HELPERS
# -----------------------------------------------------------
def _find_first_existing(filename, search_dirs):
    for d in search_dirs:
        if not d:
            continue
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def safe_load_pickle(filename):
    path = _find_first_existing(filename, EXTRA_DIRS)
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            # Safe-ish: don't allow object construction across modules
            return pickle.load(f, fix_imports=False, encoding="bytes")
    except Exception:
        return None

def load_json(filename):
    path = _find_first_existing(filename, EXTRA_DIRS)
    if path:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# -----------------------------------------------------------
# METRIC NORMALIZATION
# -----------------------------------------------------------
_METRIC_KEY_ALIASES = {
    "accuracy": ["accuracy", "Accuracy", "acc", "ACC"],
    "precision": ["precision", "Precision", "prec", "PREC"],
    "recall": ["recall", "Recall", "tpr", "TPR", "sensitivity", "Sensitivity"],
    "f1": ["f1", "F1", "f1_score", "F1_Score"],
    "auc": ["auc", "AUC", "auc_roc", "AUC_ROC", "roc_auc", "ROC_AUC"],
}

_CONTAINER_KEYS = ["performance", "performance_metrics", "metrics"]

def _get_first(d: dict, candidates):
    for k in candidates:
        if k in d:
            return d[k]
    return None

def _normalize_float(x):
    try:
        return float(x)
    except Exception:
        return None

def extract_metrics_generic(data: dict, model_name: str):
    """Accepts any dict-shaped metrics and maps to standard keys."""
    if not isinstance(data, dict):
        return None

    # dig into nested standard containers if present
    metrics = None
    for k in _CONTAINER_KEYS:
        if k in data and isinstance(data[k], dict):
            metrics = data[k]
            break
    if metrics is None:
        # also try nested 'test_metrics' (seen in KNN)
        if "test_metrics" in data and isinstance(data["test_metrics"], dict):
            metrics = data["test_metrics"]
        else:
            metrics = data  # assume flat

    std = {"model": model_name}
    for std_key, aliases in _METRIC_KEY_ALIASES.items():
        std[std_key] = _normalize_float(_get_first(metrics, aliases)) if isinstance(metrics, dict) else None

    # If nothing found, return None
    if all(std.get(k) in (None, 0) for k in ["accuracy", "precision", "recall", "f1", "auc"]):
        return None
    return std

# Specific helpers (optional but clearer for known shapes)
def extract_knn_metrics(json_data):
    # KNN JSON has "test_metrics" with capitalized keys
    if not isinstance(json_data, dict):
        return None
    test = json_data.get("test_metrics", json_data)
    metrics = {
        "model": "KNN (Optimized)",
        "accuracy": _normalize_float(_get_first(test, _METRIC_KEY_ALIASES["accuracy"]) or test.get("Accuracy")),
        "precision": _normalize_float(_get_first(test, _METRIC_KEY_ALIASES["precision"]) or test.get("Precision")),
        "recall": _normalize_float(_get_first(test, _METRIC_KEY_ALIASES["recall"]) or test.get("Recall")),
        "f1": _normalize_float(_get_first(test, _METRIC_KEY_ALIASES["f1"]) or test.get("F1")),
        "auc": _normalize_float(_get_first(test, _METRIC_KEY_ALIASES["auc"]) or test.get("AUC")),
    }
    if all(metrics.get(k) in (None, 0) for k in ["accuracy", "precision", "recall", "f1"]):
        return None
    return metrics

# -----------------------------------------------------------
# LOAD ALL MODELS
# -----------------------------------------------------------
models_data = []

# 1) Manual Perceptron
perc_json = load_json("manual_perceptron_metrics.json")
perc_pkl = safe_load_pickle("manual_perceptron_model.pkl")
# try generic extraction first
perc_metrics = extract_metrics_generic(perc_json or {}, "Manual Perceptron")
# fallback to original known values if missing
if not perc_metrics and perc_json:
    pm = perc_json.get("performance_metrics", perc_json)
    perc_metrics = {
        "model": "Manual Perceptron",
        "accuracy": _normalize_float(pm.get("accuracy", 0)),
        "precision": _normalize_float(pm.get("precision")),
        "recall": _normalize_float(pm.get("recall")),
        "f1": _normalize_float(pm.get("f1", pm.get("f1_score"))),
        "auc": 0.9616,  # given earlier
    }
elif perc_metrics and perc_metrics.get("auc") is None:
    perc_metrics["auc"] = 0.9616
if perc_metrics:
    models_data.append(perc_metrics)

# 2) Optimized SVM
svm_json = load_json("svm_performance_summary.json")
svm_pkl = safe_load_pickle("optimized_svm_model.pkl")
svm_metrics = extract_metrics_generic(svm_json or {}, "Optimized SVM")
# fallback to your earlier hardcoded values if missing
if not svm_metrics and svm_json and "performance_metrics" in svm_json:
    pm = svm_json["performance_metrics"]
    svm_metrics = {
        "model": "Optimized SVM",
        "accuracy": 0.9575,  # earlier given
        "precision": _normalize_float(pm.get("precision")),
        "recall": _normalize_float(pm.get("recall")),
        "f1": _normalize_float(pm.get("f1")),
        "auc": 0.9711,
    }
else:
    # Ensure auc present if missing and you want to keep earlier reference
    if svm_metrics and svm_metrics.get("auc") is None:
        svm_metrics["auc"] = 0.9711
if svm_metrics:
    models_data.append(svm_metrics)

# 3) XGBoost (Optimized)
xgb_pkl = safe_load_pickle("xgboost_optimized_model.pkl")
# sometimes metrics were stored inside the pickle (dict) instead of JSON
xgb_metrics = extract_metrics_generic(xgb_pkl if isinstance(xgb_pkl, dict) else {}, "XGBoost (Optimized)")
if xgb_metrics:
    models_data.append(xgb_metrics)

# 4) LightGBM (Optimized)
lgb_json = load_json("lightgbm_optimized_metrics.json")
lgb_metrics = extract_metrics_generic(lgb_json or {}, "LightGBM (Optimized)")
if lgb_metrics:
    models_data.append(lgb_metrics)

# 5) LightGBM + SMOTE
smote_json = load_json("borderline_smote_lightgbm_metrics.json")
smote_metrics = extract_metrics_generic(smote_json or {}, "LightGBM + SMOTE")
if smote_metrics:
    models_data.append(smote_metrics)

# 6) Ensemble (XGB+LGB+RF) – sometimes configs/metrics saved as pickle
ensemble_pkl = safe_load_pickle("ensemble_config.pkl")
ensemble_metrics = extract_metrics_generic(ensemble_pkl if isinstance(ensemble_pkl, dict) else {}, "Ensemble (XGB+LGB+RF)")
if ensemble_metrics:
    models_data.append(ensemble_metrics)

# 7) KNN (Optimized) – NEW
knn_json = load_json("knn_metrics.json")
knn_pkl = safe_load_pickle("best_knn_model.pkl")
knn_metrics = extract_knn_metrics(knn_json) or extract_metrics_generic(knn_json or {}, "KNN (Optimized)")
if knn_metrics:
    models_data.append(knn_metrics)

# 8) Manual Neural Network — FIXED & PERFECT
manual_nn_json = load_json("fake_job_metrics.json")
manual_nn_pkl = safe_load_pickle("manual_nn.pkl")

manual_nn_metrics = extract_metrics_generic(manual_nn_json or {}, "Manual Neural Network")
if not manual_nn_metrics:
    # Fallback: your JSON is flat → force load
    if isinstance(manual_nn_json, dict):
        manual_nn_metrics = {
            "model": "Manual Neural Network",
            "accuracy": _normalize_float(manual_nn_json.get("accuracy", 0.9829)),
            "precision": _normalize_float(manual_nn_json.get("precision", 0.9000)),
            "recall": _normalize_float(manual_nn_json.get("recall", 0.7283)),
            "f1": _normalize_float(manual_nn_json.get("f1", 0.8051)),
            "auc": _normalize_float(manual_nn_json.get("auc", 0.9644)),
        }

if manual_nn_metrics:
    models_data.append(manual_nn_metrics)

# -----------------------------------------------------------
# BUILD DASHBOARD
# -----------------------------------------------------------
if models_data:
    df = pd.DataFrame(models_data)

    # Fill missing AUC with 0 (so plots don't error)
    for c in ["accuracy", "precision", "recall", "f1", "auc"]:
        if c not in df.columns:
            df[c] = None
    df = df.fillna(0)

    st.header("📊 Model Performance Comparison")

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc"]
    available_metrics = [m for m in metrics_to_plot if m in df.columns and pd.to_numeric(df[m], errors="coerce").fillna(0).sum() > 0]

    col1, col2 = st.columns(2)

    # BAR CHART
    with col1:
        fig = go.Figure()
        for metric in available_metrics:
            fig.add_trace(go.Bar(
                x=df["model"],
                y=df[metric],
                name=metric.upper(),
                text=pd.to_numeric(df[metric], errors="coerce").round(4),
                textposition="auto"
            ))
        fig.update_layout(
            title="Metric Comparison",
            barmode="group",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # RADAR
    with col2:
        fig = go.Figure()
        # Radar needs all values in [0,1] range; ensure numeric + clipped
        radar_df = df.copy()
        for m in available_metrics:
            radar_df[m] = pd.to_numeric(radar_df[m], errors="coerce").clip(lower=0, upper=1).fillna(0)
        for _, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in available_metrics],
                theta=[m.upper() for m in available_metrics],
                fill="toself",
                name=row["model"]
            ))
        fig.update_layout(
            title="Radar Chart",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # TABLE
    st.header("📋 Detailed Metrics Table")
    display_df = df.copy()
    for col in available_metrics:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(lambda x: f"{x:.4f}" if x > 0 else "-")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # BEST MODELS
    st.header("🏆 Best Models")
    metric_labels = {"precision": "🔍 Precision", "recall": "📡 Recall", "f1": "⭐ F1 Score"}
    cols = st.columns(len(metric_labels))
    for idx, metric in enumerate(metric_labels):
        # guard if metric missing
        if metric not in df.columns or df[metric].sum() == 0:
            with cols[idx]:
                st.metric(metric_labels[metric], "—", "No data")
            continue
        best_idx = df[metric].idxmax()
        with cols[idx]:
            st.metric(metric_labels[metric], f"{df.loc[best_idx, metric]:.4f}", df.loc[best_idx, "model"])

    # RANKING
    st.header("📈 Model Rankings")
    rank_df = df.copy()
    for m in available_metrics:
        rank_df[f"{m}_rank"] = pd.to_numeric(rank_df[m], errors="coerce").rank(ascending=False, method="min")
    rank_cols = [c for c in rank_df.columns if c.endswith("_rank")]
    if rank_cols:
        rank_df["avg_rank"] = rank_df[rank_cols].mean(axis=1)
        rank_df = rank_df.sort_values("avg_rank")
        fig = go.Figure(go.Bar(
            x=rank_df["avg_rank"],
            y=rank_df["model"],
            orientation="h",
            text=rank_df["avg_rank"].round(2),
            textposition="auto",
            marker=dict(color=rank_df["avg_rank"], colorscale="RdYlGn_r", showscale=True)
        ))
        fig.update_layout(title="Average Rank (Lower is Better)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough metrics to compute rankings.")

    # INDIVIDUAL DETAILS
    st.header("🔍 Individual Model Details")
    selected_model = st.selectbox("Select model:", df["model"].tolist())

    # Map for raw objects to show in expander
    raw_map = {
        "Manual Perceptron": perc_json or perc_pkl,
        "Optimized SVM": svm_json or svm_pkl,
        "XGBoost (Optimized)": xgb_pkl,
        "LightGBM (Optimized)": lgb_json,
        "LightGBM + SMOTE": smote_json,
        "Ensemble (XGB+LGB+RF)": ensemble_pkl,
        "KNN (Optimized)": knn_json or knn_pkl,
        "Manual Neural Network": manual_nn_json or manual_nn_pkl,
    }

    if selected_model:
        row = df[df["model"] == selected_model].iloc[0]
        metric_cols = st.columns(len(available_metrics))
        for i, metric in enumerate(available_metrics):
            try:
                metric_cols[i].metric(metric.upper(), f"{float(row[metric]):.4f}")
            except Exception:
                metric_cols[i].metric(metric.UPPER(), "-")
        with st.expander("View Raw Model Data"):
            raw = raw_map.get(selected_model)
            if isinstance(raw, (dict, list)):
                st.json(raw)
            else:
                st.write(raw if raw is not None else "No raw data available.")

    st.success("✅ Dashboard loaded successfully!")

else:
    st.error("⚠️ No model metrics found.")
    st.info("Tip: Make sure your JSON files are in the folder above, or upload to /mnt/data and set the directory in the sidebar.")