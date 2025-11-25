import streamlit as st


def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

load_css()

# ------------------------------------
# HEADER SECTION
# ------------------------------------
st.markdown("""
<h1 style='font-size:48px; font-weight:700; margin-bottom:0; letter-spacing:-1px;'>
🕵️‍♂️ Fake Job Detector Dashboard
</h1>
<p style='font-size:20px; color:#555; margin-top:5px;'>
</p>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ------------------------------------
# INFO CARDS
# ------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("📊 Exploratory Data Analysis")
    st.write("""
Dive deep into the dataset with histograms, boxplots, heatmaps,
word clouds, and fraud rate comparisons across categories.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Preprocessing Pipeline")
    st.write("""
View the full preprocessing log including text cleaning,
TF-IDF vectorization, feature engineering, scaling,
and ANOVA-based feature selection.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("🤖 Model Results & Benchmarks")
    st.write("""
Explore performance metrics for SVM, Ensemble Models,
XGBoost, LightGBM, and manual Perceptron.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("🔮 Fraud Prediction")
    st.write("""
Upload data or use sample inputs to generate real-time
fraud probability predictions using your trained models.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------
# FOOTER
# ------------------------------------
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)


