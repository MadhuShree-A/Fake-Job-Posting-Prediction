import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from scipy.sparse import csr_matrix, hstack
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

st.title("🔮 Fake Job Posting Prediction")
st.write("Upload a CSV file — preprocessing + prediction will run automatically.")

# -------------------------------------------------
# ✅ Define ManualPerceptron class FIRST (BEFORE loading models)
# -------------------------------------------------
class ManualPerceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, class_weight=None):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.class_weight = class_weight
        self.weights = None
        self.errors_history = []
        self.loss_history = []

    def _calculate_sample_weights(self, y):
        from sklearn.utils.class_weight import compute_class_weight
        if self.class_weight == 'balanced':
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            sample_weights = np.where(y == 1, class_weights[1], class_weights[0])
        elif isinstance(self.class_weight, dict):
            sample_weights = np.where(y == 1, self.class_weight[1], self.class_weight[0])
        else:
            sample_weights = np.ones(len(y))
        return sample_weights

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        sample_weights = self._calculate_sample_weights(y)

        for epoch in range(self.n_iters):
            total_errors = 0
            total_loss = 0

            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights)
                y_pred = self.activation(linear_output)

                error = (y[i] - y_pred) * sample_weights[i]
                total_loss += abs(error)

                if error != 0:
                    self.weights += self.lr * error * X[i]
                    total_errors += 1

            self.errors_history.append(total_errors)
            self.loss_history.append(total_loss / n_samples)

            if total_errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        return self.activation(linear_output)

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights)
        probabilities = 1 / (1 + np.exp(-linear_output))
        return probabilities

    def decision_function(self, X):
        return np.dot(X, self.weights)
    
# -------------------------------------------------
# ✅ Define ManualKNN class (for KNN model loading)
# -------------------------------------------------
class ManualKNN:
    def __init__(self, n_neighbors=3, p=1, weights="distance"):
        self.k = n_neighbors
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).astype(int)

    def compute_distances(self, x):
        return np.sum(np.abs(self.X_train - x), axis=1)

    def predict_one(self, x):
        distances = self.compute_distances(x)
        idx = np.argsort(distances)[:self.k]
        labels = self.y_train[idx]
        dists = distances[idx]
        weights = 1 / (dists + 1e-10)
        score1 = np.sum(weights * labels)
        score0 = np.sum(weights * (1 - labels))
        return 1 if score1 > score0 else 0

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self.predict_one(row) for row in X])

    def predict_proba(self, X):
        X = np.asarray(X)
        probs = []
        for row in X:
            distances = self.compute_distances(row)
            idx = np.argsort(distances)[:self.k]
            labels = self.y_train[idx]
            dists = distances[idx]
            weights = 1 / (dists + 1e-10)
            p1 = np.sum(weights * labels)
            p0 = np.sum(weights * (1 - labels))
            probs.append([p0/(p0+p1), p1/(p0+p1)])
        return np.array(probs)

# -------------------------------------------------
# ✅ Manual Neural Network Class (MUST MATCH TRAINING EXACTLY)
# -------------------------------------------------
class ManualL3NN:
    def __init__(self, input_dim, hidden_dims=[256,128,64], output_dim=1):
        np.random.seed(42)
        self.params = {}
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(1, len(dims)):
            self.params[f"W{i}"] = np.random.randn(dims[i-1], dims[i]) * np.sqrt(2/dims[i-1])
            self.params[f"b{i}"] = np.zeros((1, dims[i]))
            if i < len(dims)-1:
                self.params[f"gamma{i}"] = np.ones((1, dims[i]))
                self.params[f"beta{i}"] = np.zeros((1, dims[i]))
        self.bn_params = {i: {'running_mean': np.zeros((1, hidden_dims[i-1])),
                              'running_var': np.ones((1, hidden_dims[i-1]))}
                          for i in range(1, len(hidden_dims)+1)}
        self.m, self.v, self.t = {}, {}, 0
        for k in self.params:
            self.m[k], self.v[k] = np.zeros_like(self.params[k]), np.zeros_like(self.params[k])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def batchnorm_forward(self, z, layer, mode):
        gamma, beta = self.params[f"gamma{layer}"], self.params[f"beta{layer}"]
        bn = self.bn_params[layer]
        if mode=="train":
            mu = np.mean(z, axis=0, keepdims=True)
            var = np.var(z, axis=0, keepdims=True)
            var = np.clip(var,1e-8,None)
            z_hat = (z - mu)/np.sqrt(var+1e-8)
            out = gamma*z_hat + beta
            bn["running_mean"] = 0.9*bn["running_mean"] + 0.1*mu
            bn["running_var"] = 0.9*bn["running_var"] + 0.1*var
            return out, (z, z_hat, mu, var, gamma, beta)
        else:
            z_hat = (z - bn["running_mean"])/np.sqrt(bn["running_var"]+1e-8)
            return gamma*z_hat + beta, None

    def batchnorm_backward(self, dout, cache):
        z, z_hat, mu, var, gamma, beta = cache
        N = dout.shape[0]
        dbeta = np.sum(dout, axis=0, keepdims=True)
        dgamma = np.sum(dout*z_hat, axis=0, keepdims=True)
        dz_hat = dout*gamma
        var = np.clip(var,1e-8,None)
        std_inv = 1./np.sqrt(var+1e-8)
        dvar = np.sum(dz_hat*(z-mu)*-0.5*np.power(var+1e-8,-1.5), axis=0, keepdims=True)
        dmu = np.sum(dz_hat*-std_inv, axis=0, keepdims=True) + dvar*np.mean(-2*(z-mu), axis=0, keepdims=True)
        dz = dz_hat*std_inv + dvar*2*(z-mu)/N + dmu/N
        return dz, dgamma, dbeta

    def forward(self, X, mode="eval", dropout_rate=0.0):
        caches = {}; activations=[X]
        for i in range(1,4):
            A_prev = activations[-1]
            Z = A_prev @ self.params[f"W{i}"] + self.params[f"b{i}"]
            Z_bn, bn_cache = self.batchnorm_forward(Z, i, mode)
            A = self.relu(Z_bn)
            activations.append(A)
            caches[i]=(Z, Z_bn, bn_cache)
        Z_out = activations[-1] @ self.params["W4"] + self.params["b4"]
        y_hat = self.sigmoid(Z_out)
        caches["activations"]=activations
        return y_hat, caches

    def predict_proba(self, X):
        y_pred,_ = self.forward(X, mode="eval")
        return y_pred.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)>=threshold).astype(int)

# -------------------------------------------------
# ✅ Utility: scores → probabilities (sigmoid)
# -------------------------------------------------
def scores_to_probabilities(scores):
    """Convert decision scores to probabilities using sigmoid"""
    return 1 / (1 + np.exp(-scores))

# -------------------------------------------------
# ✅ Load Artifacts
# -------------------------------------------------
ART = "artifacts"
MODELS = "models"
EXTRA_DIRS = [MODELS, "/mnt/data"]  # search order (local folder and uploaded files area)

def _find_first_existing(filename, search_dirs):
    for d in search_dirs:
        if not d:
            continue
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def safe_pickle_load(filename):
    """Find file in EXTRA_DIRS and load pickle safely."""
    path = _find_first_existing(filename, EXTRA_DIRS)
    if not path:
        raise FileNotFoundError(f"File not found in {EXTRA_DIRS}: {filename}")
    with open(path, "rb") as f:
        # use safe defaults; if old pickles need class resolution, ensure classes defined above
        return pickle.load(f, fix_imports=False, encoding="bytes")

try:
    tfidf = pickle.load(open(os.path.join(ART, "tfidf_vectorizer.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(ART, "feature_scaler.pkl"), "rb"))
    selector = pickle.load(open(os.path.join(ART, "feature_selector.pkl"), "rb"))
    ohe_columns = pickle.load(open(os.path.join(ART, "ohe_columns.pkl"), "rb"))
    numeric_feature_names = pickle.load(open(os.path.join(ART, "numeric_feature_names.pkl"), "rb"))
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# Download NLTK data
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    pass

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# ✅ Required Columns
# -------------------------------------------------
TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["location", "department", "employment_type", "required_experience",
            "required_education", "industry", "function"]
BINARY_COLS = ["telecommuting", "has_company_logo", "has_questions"]

# -------------------------------------------------
# ✅ Helper Functions
# -------------------------------------------------
def clean_text(txt):
    txt = str(txt)
    soup = BeautifulSoup(txt, "html.parser")
    txt = soup.get_text()
    txt = re.sub(r"[^a-zA-Z\s]", "", txt).lower()
    words = [lemmatizer.lemmatize(w) for w in txt.split() if w not in stop_words]
    return " ".join(words) if words else "emptytext"

def reduce_cardinality(series, threshold=50, other_label='Other'):
    value_counts = series.value_counts()
    keep = value_counts[value_counts >= threshold].index
    return series.where(series.isin(keep), other_label)

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
def _has_estimator_api(x):
    return hasattr(x, "predict") or hasattr(x, "predict_proba") or hasattr(x, "decision_function")

def _normalize_keys(d):
    # convert byte-string keys (b'key') to str ('key') for pickles saved with encoding="bytes"
    return { (k.decode() if isinstance(k, (bytes, bytearray)) else k): v for k, v in d.items() }

def _unwrap_model(obj):
    """
    Recursively unwrap:
    - 0-D numpy arrays
    - dict wrappers (byte or string keys)
    - nested arrays with estimators
    - return the first estimator-like object found
    """

    # ✅ 1. Unwrap top-level 0-D numpy array
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return _unwrap_model(obj.item())   # 🔥 recursive

    # ✅ 2. If already estimator → return immediately
    if _has_estimator_api(obj):
        return obj

    # ✅ 3. If dict → normalize + scan keys and values
    if isinstance(obj, dict):
        d = _normalize_keys(obj)

        # common keys
        candidate = [
            "best_model", "model", "best_estimator_", "estimator",
            "clf", "classifier", "rf", "knn", "lgb", "xgb"
        ]

        # first check standard keys
        for k in candidate:
            if k in d:
                extracted = _unwrap_model(d[k])  # 🔥 recursive unwrap
                if _has_estimator_api(extracted):
                    return extracted

        # then scan dict values
        for v in d.values():
            extracted = _unwrap_model(v)  # 🔥 recursive unwrap
            if _has_estimator_api(extracted):
                return extracted

        return obj  # no estimator inside

    # ✅ 4. If array with items → scan elements
    if isinstance(obj, np.ndarray) and obj.ndim > 0:
        for v in obj.ravel():
            extracted = _unwrap_model(v)  # 🔥 recursive unwrap
            if _has_estimator_api(extracted):
                return extracted

    # ✅ Nothing found
    return obj

# -------------------------------------------------
# ✅ FLEXIBLE Manual Neural Network Class 
# -------------------------------------------------
class ManualL3NN:
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=1):
        np.random.seed(42)
        self.params = {}
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(1, len(dims)):
            self.params[f"W{i}"] = np.random.randn(dims[i-1], dims[i]) * np.sqrt(2/dims[i-1])
            self.params[f"b{i}"] = np.zeros((1, dims[i]))
            if i < len(dims)-1:
                self.params[f"gamma{i}"] = np.ones((1, dims[i]))
                self.params[f"beta{i}"] = np.zeros((1, dims[i]))

        self.bn_params = {
            i: {
                "running_mean": np.zeros((1, hidden_dims[i-1])),
                "running_var": np.ones((1, hidden_dims[i-1]))
            }
            for i in range(1, len(hidden_dims)+1)
        }

        self.m, self.v, self.t = {}, {}, 0
        for k in self.params:
            self.m[k] = np.zeros_like(self.params[k])
            self.v[k] = np.zeros_like(self.params[k])

    def sigmoid(self, z):
        return 1/(1 + np.exp(-np.clip(z, -20, 20)))

    def relu(self, z):
        return np.maximum(0, z)

    def batchnorm_forward(self, z, layer, mode):
        gamma = self.params[f"gamma{layer}"]
        beta = self.params[f"beta{layer}"]
        bn = self.bn_params[layer]

        if mode == "train":
            mu = np.mean(z, axis=0, keepdims=True)
            var = np.var(z, axis=0, keepdims=True)
            var = np.clip(var, 1e-8, None)
            z_hat = (z - mu)/np.sqrt(var + 1e-8)
            out = gamma * z_hat + beta
            bn["running_mean"] = 0.9 * bn["running_mean"] + 0.1 * mu
            bn["running_var"] = 0.9 * bn["running_var"] + 0.1 * var
            return out, (z, z_hat, mu, var, gamma2, beta)
        else:
            z_hat = (z - bn["running_mean"])/np.sqrt(bn["running_var"] + 1e-8)
            return gamma * z_hat + beta, None

    def forward(self, X, mode="eval"):
        caches = {}
        activations = [X]
        for i in range(1, 4):
            A_prev = activations[-1]
            Z = A_prev @ self.params[f"W{i}"] + self.params[f"b{i}"]
            Z_bn, bn_cache = self.batchnorm_forward(Z, i, mode)
            A = self.relu(Z_bn)
            activations.append(A)
            caches[i] = (Z, Z_bn, bn_cache)
        Z_out = activations[-1] @ self.params["W4"] + self.params["b4"]
        y_hat = self.sigmoid(Z_out)
        caches["activations"] = activations
        return y_hat, caches

    def predict_proba(self, X):
        y_pred, _ = self.forward(X, mode="eval")
        return y_pred.ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
# -------------------------------------------------
# ✅ Model Prediction Loader (supports multiple shapes)
# -------------------------------------------------
def predict_with_model(model_file, X):
    """
    Loads the requested model (searching in models/ and /mnt/data),
    handles different pickle structures, and returns (preds, probs, err).
    """
    try:
        # ✅ expose helpers & custom classes to pickle
        import __main__
        __main__.ManualPerceptron = ManualPerceptron
        __main__.scores_to_probabilities = scores_to_probabilities
        __main__.ManualPerceptron = ManualPerceptron
        __main__.ManualKNN = ManualKNN
        __main__.ManualL3NN = ManualL3NN

        # ✅ load the raw object
        raw = safe_pickle_load(model_file)

        # ===== Manual Perceptron =====
        if model_file == "manual_perceptron_model.pkl":
            model = _unwrap_model(raw)
            if not _has_estimator_api(model):
                return None, None, "ManualPerceptron not valid"

            # ADD BIAS AUTOMATICALLY HERE – NO MATTER WHAT X YOU GAVE
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            X_with_bias = np.hstack([np.ones((X_dense.shape[0], 1)), X_dense])

            probs = model.predict_proba(X_with_bias)
            preds = (probs >= 0.5).astype(int)
            return preds, probs, None
        # ===== Optimized SVM =====
        if model_file == "optimized_svm_model.pkl":
            model = _unwrap_model(raw)
            if not _has_estimator_api(model):
                return None, None, "❌ Invalid SVM model structure (no estimator found)."
            th = 0.0
            if isinstance(raw, dict):
                d = _normalize_keys(raw)
                th = d.get("optimal_threshold", 0.0)

            if hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                probs = scores_to_probabilities(scores)
                preds = (scores > th).astype(int)
                return preds, probs, None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                preds = (probs > th if th else probs > 0.5).astype(int)
                return preds, probs, None
            return None, None, "❌ Invalid SVM model: no decision_function/predict_proba."

        # ===== LightGBM & SMOTE variants =====
        if ("lightgbm" in model_file) or ("smote" in model_file):
            model = _unwrap_model(raw)
            if not _has_estimator_api(model):
                return None, None, "❌ LightGBM pickle doesn't contain a valid estimator."
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                probs = scores_to_probabilities(model.decision_function(X))
            else:
                return None, None, "❌ Unknown LightGBM model."
            preds = (probs >= 0.5).astype(int)
            return preds, probs, None

        # ===== Ensemble Config =====
        if model_file == "ensemble_config.pkl":
            if not isinstance(raw, dict):
                return None, None, "❌ Invalid ensemble config."
            cfg = _normalize_keys(raw)

            try:
                xgb = _unwrap_model(safe_pickle_load("xgb_fraud_model.pkl"))
                lgb = _unwrap_model(safe_pickle_load("lgb_fraud_model.pkl"))
                rf  = _unwrap_model(safe_pickle_load("rf_fraud_model.pkl"))

                for mname, m in [("XGB", xgb), ("LGB", lgb), ("RF", rf)]:
                    if not _has_estimator_api(m):
                        return None, None, f"❌ Ensemble component {mname} is not a valid estimator."

                def proba(m):
                    if hasattr(m, "predict_proba"):
                        return m.predict_proba(X)[:, 1]
                    if hasattr(m, "decision_function"):
                        return scores_to_probabilities(m.decision_function(X))
                    raise ValueError("No proba/decision")

                probs = 0.4*proba(xgb) + 0.4*proba(lgb) + 0.2*proba(rf)
                th = cfg.get("optimal_threshold", 0.5)
                preds = (probs >= th).astype(int)
                return preds, probs, None

            except Exception as e:
                return None, None, f"❌ Error loading ensemble components: {e}"

                # ===== KNN (Optimized) - SPECIAL HANDLING =====
        if model_file == "best_knn_model.pkl":
            # Your saved file contains: {"svd": svd, "knn": knn}
            if isinstance(raw, dict):
                try:
                    svd = raw["svd"]
                    knn = raw["knn"]
                except KeyError:
                    return None, None, "❌ best_knn_model.pkl is missing 'svd' or 'knn' key"
            else:
                # Old format: only knn object saved → try to load SVD separately
                try:
                    svd = safe_pickle_load("svd_component.pkl")  # ← you need this file
                    knn = raw
                except:
                    return None, None, "❌ Old KNN format detected. Please re-save with SVD or provide svd_component.pkl"

            # Apply exact same pipeline as training
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            X_svd = svd.transform(X_dense)
            X_norm = X_svd / (np.linalg.norm(X_svd, axis=1, keepdims=True) + 1e-10)
            
            preds = knn.predict(X_norm)
            probs = knn.predict_proba(X_norm)[:, 1]
            return preds, probs, None

  
                # Manual Neural Network – FINAL VERSION (GUARANTEED TO DETECT FRAUD)
                # Manual Neural Network – CLEAN VERSION (no top-5 print)
        if model_file == "manual_nn.pkl":
            try:
                X = X_final.toarray().astype(np.float32)
                model = _unwrap_model(raw)
                probs = model.predict_proba(X)
                preds = (probs >= 0.5).astype(int)    # sacred threshold
                return preds, probs, None
            except Exception as e:
                return None, None, f"NN Error: {e}"
        

                # ===== Logistic Regression =====
        if model_file == "LogisticRegression_best.pkl":
            model = _unwrap_model(raw)
            if not _has_estimator_api(model):
                return None, None, "❌ Logistic Regression pickle invalid"

            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
            return preds, probs, None



        # ===== Default Fallback =====
        model = _unwrap_model(raw)
        if not _has_estimator_api(model):
            if isinstance(raw, dict):
                return None, None, f"❌ Unknown model type; dict keys: {list(_normalize_keys(raw).keys())}"
            return None, None, "❌ Unknown model type."
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= 0.5).astype(int)
            return preds, probs, None
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            probs = scores_to_probabilities(scores)
            preds = (scores > 0).astype(int)
            return preds, probs, None

        return None, None, "❌ Unknown model type."

    except FileNotFoundError as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        return None, None, f"❌ Error loading model: {e}"


# -------------------------------------------------
# ✅ File Upload
# -------------------------------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write(f"📊 Uploaded {len(df)} records")

        # Fill Missing Values
        for col in TEXT_COLS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)

        for col in BINARY_COLS:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        if "salary_range" not in df.columns:
            df["salary_range"] = ""

        # Apply Cardinality Reduction
        df['location'] = reduce_cardinality(df['location'], 100)
        df['department'] = reduce_cardinality(df['department'], 50)
        df['industry'] = reduce_cardinality(df['industry'], 30)

        df["salary_specified"] = df["salary_range"].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
        )

        # Text Processing
        df["text_raw"] = df[TEXT_COLS].agg(' '.join, axis=1)
        df["text_clean"] = df["text_raw"].apply(clean_text)

        # Extract Numeric Features
        df_num = df.apply(extract_text_features, axis=1)

        continuous_cols = [
            'char_count', 'word_count', 'unique_words', 'avg_word_len',
            'all_caps_ratio', 'uppercase_word_count', 'text_richness'
        ]
        binary_extra = ['num_exclaims', 'num_questions', 'has_email', 'has_url']

        X_cont = scaler.transform(df_num[continuous_cols])
        X_bin_extra = df_num[binary_extra].values
        X_bin_orig = df[BINARY_COLS].values
        X_salary = df[['salary_specified']].values
        X_num = np.hstack([X_cont, X_bin_extra, X_bin_orig, X_salary])

        # TF-IDF Vectorization
        X_text = tfidf.transform(df["text_clean"])

        # Categorical Encoding
        st.write("🔄 Processing categorical features...")

        df_cat = pd.get_dummies(df[CAT_COLS], drop_first=True)

        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype(bool).astype(int)

        df_cat_aligned = pd.DataFrame(0, index=df_cat.index, columns=ohe_columns, dtype=np.int32)

        for col in df_cat.columns:
            if col in ohe_columns:
                df_cat_aligned[col] = df_cat[col].values

        cat_array = df_cat_aligned.values.astype(np.float64)

        # Combine Features
        X_combined = hstack([
            X_text,
            csr_matrix(X_num.astype(np.float64)),
            csr_matrix(cat_array)
        ], format="csr")

                # Feature Selection
               
        try:
            X_final = selector.transform(X_combined)
            st.success(f"Using SelectKBest → {X_final.shape}")
        except:
            # Extra safety for NN
            if choice == "Manual Neural Network":
                if not hasattr(selector, 'transform') or X_final.shape[1] != 500:
                    st.error("Manual NN requires exactly 500 selected features. Selector failed or wrong k.")
                    st.stop()
        # -------------------------------------------------
        # ✅ Model Selection & Prediction (now includes KNN + RF)
        # -------------------------------------------------
        st.subheader("Select a Model")

        choice = st.selectbox(
            "Choose model",
            [
                "Manual Perceptron",
                "Optimized SVM",
                "LightGBM (Optimized)",
                "Borderline SMOTE + LGBM",
                "KNN (Optimized)",           
                "Ensemble (XGB + LGB + RF)",
                "Manual Neural Network",     
            ],
            key="selected_model"  
        )

        
        MODEL_MAP = {
            "Manual Perceptron": "manual_perceptron_model.pkl",
            "Optimized SVM": "optimized_svm_model.pkl",
            "LightGBM (Optimized)": "lightgbm_optimized_model.pkl",
            "Borderline SMOTE + LGBM": "borderline_smote_lightgbm_model.pkl",
            "KNN (Optimized)": "best_knn_model.pkl",            
            "Ensemble (XGB + LGB + RF)": "ensemble_config.pkl",
            "Manual Neural Network": "manual_nn.pkl",             
                    
        }

        model_file = MODEL_MAP[choice]

        with st.spinner("Running prediction..."):
            preds, probs, err = predict_with_model(model_file, X_final)

        if err:
            st.error(err)
            if "SVM" in choice:
                st.info("💡 Try using a different model like LightGBM, Random Forest, or KNN.")
        else:
            st.success("✅ Prediction Complete!")

            out = df.copy()
            out["prediction"] = np.asarray(preds).flatten().astype(int)
            out["fraud_probability"] = np.asarray(probs).astype(float)
            out["result"] = out["prediction"].map({0: "✅ Legitimate", 1: "⚠️ FRAUD"})

            fraud_count = int(out["prediction"].sum())
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Jobs", len(out))
            with col2:
                st.metric("Fraudulent", fraud_count, delta=f"{fraud_count/len(out)*100:.1f}%", delta_color="inverse")
            with col3:
                st.metric("Legitimate", len(out) - fraud_count)

            st.subheader("Prediction Results")
            display_cols = ["title", "location", "result", "fraud_probability"]
            # Style only the 'result' column red when FRAUD
            styled = out[display_cols].style.apply(
                lambda s: ['background-color: #ffcccc' if isinstance(x, str) and 'FRAUD' in x else '' for x in s],
                subset=['result']
            )
            st.dataframe(styled, use_container_width=True)

            st.download_button(
                "📥 Download Full Predictions",
                out.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error during preprocessing: {e}")
        import traceback
        st.code(traceback.format_exc())