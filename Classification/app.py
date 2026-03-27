# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="centered"
)

# ============================================================
# Load Model & Encoder
# ============================================================
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)

    model_path = os.path.join(BASE_DIR, "svm_tuned.pkl")
    encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    if not os.path.exists(encoder_path):
        st.error(f"Encoder file not found: {encoder_path}")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    return model, le


# ✅ IMPORTANT: CALL FUNCTION HERE
model, le = load_artifacts()

# ============================================================
# Bean Class Info
# ============================================================
bean_info = {
    "SEKER":    {"emoji": "🟡", "desc": "Small, round, light-colored bean"},
    "BARBUNYA": {"emoji": "🔴", "desc": "Medium, speckled, kidney-shaped bean"},
    "BOMBAY":   {"emoji": "⚫", "desc": "Large, dark, round bean"},
    "CALI":     {"emoji": "🟤", "desc": "Large, elongated, brownish bean"},
    "HOROZ":    {"emoji": "🟠", "desc": "Medium, elongated, hook-shaped bean"},
    "SIRA":     {"emoji": "🟢", "desc": "Medium, oval, yellowish-green bean"},
    "DERMASON": {"emoji": "⚪", "desc": "Small, oval, white/cream bean"},
}

# ============================================================
# Header
# ============================================================
st.title("🫘 Dry Bean Classifier")
st.markdown("Enter the **physical measurements** of a bean to predict its class.")
st.markdown("---")

# ============================================================
# Input Form
# ============================================================
st.subheader("📏 Bean Measurements")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", 0.0, value=50000.0)
    perimeter = st.number_input("Perimeter", 0.0, value=900.0)
    major_axis = st.number_input("Major Axis Length", 0.0, value=350.0)
    minor_axis = st.number_input("Minor Axis Length", 0.0, value=200.0)
    aspect_ratio = st.number_input("Aspect Ratio", 0.0, value=1.75)
    eccentricity = st.number_input("Eccentricity", 0.0, 1.0, value=0.82)
    convex_area = st.number_input("Convex Area", 0.0, value=52000.0)

with col2:
    equiv_diameter = st.number_input("Equiv. Diameter", 0.0, value=250.0)
    extent = st.number_input("Extent", 0.0, 1.0, value=0.75)
    solidity = st.number_input("Solidity", 0.0, 1.0, value=0.98)
    roundness = st.number_input("Roundness", 0.0, 1.0, value=0.78)
    compactness = st.number_input("Compactness", 0.0, 1.0, value=0.75)
    shape_factor1 = st.number_input("Shape Factor 1", 0.0, value=0.006, format="%.4f")
    shape_factor2 = st.number_input("Shape Factor 2", 0.0, value=0.0018, format="%.4f")

col3, col4 = st.columns(2)
with col3:
    shape_factor3 = st.number_input("Shape Factor 3", 0.0, 1.0, value=0.64)
with col4:
    shape_factor4 = st.number_input("Shape Factor 4", 0.0, 1.0, value=0.99)

# ============================================================
# Predict Button
# ============================================================
st.markdown("---")

if st.button("🔍 Predict Bean Class", use_container_width=True):

    try:
        features = np.array([[
            area, perimeter, major_axis, minor_axis,
            aspect_ratio, eccentricity, convex_area,
            equiv_diameter, extent, solidity,
            roundness, compactness,
            shape_factor1, shape_factor2, shape_factor3, shape_factor4
        ]])

        pred_encoded = model.predict(features)
        pred_class = le.inverse_transform(pred_encoded)[0]

        pred_proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None

        st.subheader("🎯 Prediction Result")

        info = bean_info.get(pred_class, {"emoji": "🫘", "desc": "Bean variety"})
        st.success(f"{info['emoji']} Predicted Class: **{pred_class}**")
        st.info(info["desc"])

        if pred_proba is not None:
            st.subheader("📊 Class Probabilities")
            classes = le.inverse_transform(np.arange(len(pred_proba)))
            prob_df = pd.DataFrame({
                "Bean Class": classes,
                "Confidence": pred_proba
            }).sort_values("Confidence", ascending=False)

            st.dataframe(prob_df, use_container_width=True)
            st.bar_chart(prob_df.set_index("Bean Class"))

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("Dry Bean Classification using SVM")

    st.header("🫘 Bean Classes")
    for cls, meta in bean_info.items():
        st.write(f"{meta['emoji']} {cls} - {meta['desc']}")
