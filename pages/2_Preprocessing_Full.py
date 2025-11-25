import streamlit as st
import os
import streamlit as st

# Load custom CSS
def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("⚙️ Full Preprocessing Log")

log_dir = "preprocessing_logs"

if not os.path.exists(log_dir):
    st.error("❌ preprocessing_logs folder not found.")
else:
    logs = sorted(os.listdir(log_dir))

    if not logs:
        st.warning("No log files found.")
    else:
        latest_log = os.path.join(log_dir, logs[-1])
        st.success(f"Showing latest log file: {logs[-1]}")

        with open(latest_log, "r", encoding="utf-8") as f:
            log_text = f.read()

        st.code(log_text, language="bash")
