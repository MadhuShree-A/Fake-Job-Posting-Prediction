import streamlit as st
import os
from PIL import Image
import streamlit as st

def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("📊 Full EDA Report")

EDA_DIR = "eda_plots"

if not os.path.exists(EDA_DIR):
    st.error(" The folder 'eda_plots/' does not exist.")
else:
    images = sorted([
        f for f in os.listdir(EDA_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not images:
        st.warning("No EDA images found in 'eda_plots/'.")
    else:
        for img_name in images:
            st.subheader(img_name.replace(".png", "").replace("_", " "))
            img = Image.open(os.path.join(EDA_DIR, img_name))
            st.image(img)
            st.divider()
