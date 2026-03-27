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
# Load Model, Scaler & Encoder
# ============================================================
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)

    with open(os.path.join(BASE_DIR, "svm_tuned.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return model, le, scaler

model, le, scaler = load_artifacts()

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
    area            = st.number_input("Area",             min_value=0.0,  value=50000.0,  step=100.0,   help="Area of the bean zone (pixels)")
    perimeter       = st.number_input("Perimeter",        min_value=0.0,  value=900.0,    step=1.0,     help="Bean circumference (pixels)")
    major_axis      = st.number_input("Major Axis Length",min_value=0.0,  value=350.0,    step=1.0,     help="Longest diameter of the bean")
    minor_axis      = st.number_input("Minor Axis Length",min_value=0.0,  value=200.0,    step=1.0,     help="Shortest diameter of the bean")
    aspect_ratio    = st.number_input("Aspect Ratio",     min_value=0.0,  value=1.75,     step=0.01,    help="Major axis / Minor axis")
    eccentricity    = st.number_input("Eccentricity",     min_value=0.0,  max_value=1.0,  value=0.82,   step=0.01, help="Eccentricity of the ellipse (0=circle, 1=line)")
    convex_area     = st.number_input("Convex Area",      min_value=0.0,  value=52000.0,  step=100.0,   help="Smallest convex polygon area")

with col2:
    equiv_diameter  = st.number_input("Equiv. Diameter",  min_value=0.0,  value=250.0,    step=1.0,     help="Diameter of a circle with same area")
    extent          = st.number_input("Extent",           min_value=0.0,  max_value=1.0,  value=0.75,   step=0.01, help="Ratio of pixels in bounding box")
    solidity        = st.number_input("Solidity",         min_value=0.0,  max_value=1.0,  value=0.98,   step=0.001,help="Area / Convex Area ratio")
    roundness       = st.number_input("Roundness",        min_value=0.0,  max_value=1.0,  value=0.78,   step=0.01, help="Circularity measure")
    compactness     = st.number_input("Compactness",      min_value=0.0,  max_value=1.0,  value=0.75,   step=0.01, help="Roundness of the bean")
    shape_factor1   = st.number_input("Shape Factor 1",   min_value=0.0,  value=0.006,    step=0.0001,  format="%.4f", help="Major axis related shape descriptor")
    shape_factor2   = st.number_input("Shape Factor 2",   min_value=0.0,  value=0.0018,   step=0.0001,  format="%.4f", help="Minor axis related shape descriptor")

col3, col4 = st.columns(2)
with col3:
    shape_factor3   = st.number_input("Shape Factor 3",   min_value=0.0,  max_value=1.0,  value=0.64,   step=0.01, help="Compactness related descriptor")
with col4:
    shape_factor4   = st.number_input("Shape Factor 4",   min_value=0.0,  max_value=1.0,  value=0.99,   step=0.01, help="Convexity related descriptor")

# ============================================================
# Predict Button
# ============================================================
st.markdown("---")

if st.button("🔍 Predict Bean Class", use_container_width=True, type="primary"):

    # Assemble feature vector (must match training column order)
    features = np.array([[
        area, perimeter, major_axis, minor_axis,
        aspect_ratio, eccentricity, convex_area,
        equiv_diameter, extent, solidity,
        roundness, compactness,
        shape_factor1, shape_factor2, shape_factor3, shape_factor4
    ]])

    # ✅ FIX: Apply the same StandardScaler used during training
    features_scaled = scaler.transform(features)

    # Predict
    pred_encoded = model.predict(features_scaled)
    pred_proba   = model.predict_proba(features_scaled)[0] if hasattr(model, "predict_proba") else None
    pred_class   = le.inverse_transform(pred_encoded)[0]

    # ---- Result Card ----
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    info = bean_info.get(pred_class, {"emoji": "🫘", "desc": "Bean variety"})

    st.success(f"### {info['emoji']}  Predicted Class: **{pred_class}**")
    st.info(f"📖 {info['desc']}")

    # ---- Confidence Bar (if SVM with probability=True) ----
    if pred_proba is not None:
        st.markdown("#### 📊 Class Probabilities")
        classes = le.inverse_transform(np.arange(len(pred_proba)))
        prob_df = pd.DataFrame({
            "Bean Class": classes,
            "Confidence": pred_proba
        }).sort_values("Confidence", ascending=False).reset_index(drop=True)

        prob_df["Confidence %"] = (prob_df["Confidence"] * 100).round(2).astype(str) + "%"

        st.dataframe(
            prob_df[["Bean Class", "Confidence %"]],
            use_container_width=True,
            hide_index=True
        )
        st.bar_chart(prob_df.set_index("Bean Class")["Confidence"])

    # ---- Input Summary ----
    with st.expander("📋 View Input Summary"):
        input_df = pd.DataFrame({
            "Feature": [
                "Area", "Perimeter", "Major Axis Length", "Minor Axis Length",
                "Aspect Ratio", "Eccentricity", "Convex Area",
                "Equiv. Diameter", "Extent", "Solidity",
                "Roundness", "Compactness",
                "Shape Factor 1", "Shape Factor 2", "Shape Factor 3", "Shape Factor 4"
            ],
            "Value": features[0]  # show original unscaled values to the user
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ============================================================
# Sidebar — About
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts the **variety of dry bean** based on physical measurements
    extracted from images.

    **Model:** SVM (Tuned)
    **Dataset:** Dry Bean Dataset (UCI)
    **Classes:** 7 bean varieties
    """)

    st.markdown("---")
    st.header("🫘 Bean Classes")
    for cls, meta in bean_info.items():
        st.markdown(f"{meta['emoji']} **{cls}** — {meta['desc']}")

    st.markdown("---")
    st.caption("Built with Streamlit + Scikit-learn")
