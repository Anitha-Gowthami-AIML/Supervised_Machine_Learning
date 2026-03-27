"""
app.py - Dry Bean Type Classifier
Run: streamlit run app.py
Requires: best_model.pkl, label_encoder.pkl, feature_names.pkl
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, le, feature_names

model, le, feature_names = load_artifacts()

# App title
st.title("🌿 Dry Bean Type Classifier")
st.markdown("Enter the physical measurements of a bean to predict its type.")
st.markdown("**Model:** SVM (Tuned) — Best performing classifier")

st.divider()

# Feature input layout
st.subheader("Bean Physical Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area", min_value=0.0, value=28395.0, step=100.0)
    perimeter = st.number_input("Perimeter", min_value=0.0, value=610.29, step=1.0)
    major_axis = st.number_input("Major Axis Length", min_value=0.0, value=208.18, step=1.0)
    minor_axis = st.number_input("Minor Axis Length", min_value=0.0, value=173.89, step=1.0)
    aspect_ratio = st.number_input("Aspect Ratio", min_value=0.0, value=1.197, step=0.01)
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.55, step=0.01)

with col2:
    convex_area = st.number_input("Convex Area", min_value=0.0, value=28715.0, step=100.0)
    equiv_diameter = st.number_input("Equivalent Diameter", min_value=0.0, value=190.14, step=1.0)
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.764, step=0.01)
    solidity = st.number_input("Solidity", min_value=0.0, max_value=1.0, value=0.989, step=0.001)
    roundness = st.number_input("Roundness", min_value=0.0, max_value=1.0, value=0.958, step=0.01)

with col3:
    compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, value=0.913, step=0.01)
    sf1 = st.number_input("ShapeFactor1", min_value=0.0, value=0.00733, step=0.0001, format="%.5f")
    sf2 = st.number_input("ShapeFactor2", min_value=0.0, value=0.00315, step=0.0001, format="%.5f")
    sf3 = st.number_input("ShapeFactor3", min_value=0.0, max_value=1.0, value=0.834, step=0.01)
    sf4 = st.number_input("ShapeFactor4", min_value=0.0, max_value=1.0, value=0.999, step=0.001)

st.divider()

# Predict button
if st.button("🔍 Predict Bean Type", type="primary", use_container_width=True):
    input_data = np.array([[
        area, perimeter, major_axis, minor_axis, aspect_ratio,
        eccentricity, convex_area, equiv_diameter, extent, solidity,
        roundness, compactness, sf1, sf2, sf3, sf4
    ]])

    input_df = pd.DataFrame(input_data, columns=feature_names)
    prediction = model.predict(input_df)
    predicted_class = le.inverse_transform(prediction)[0]

    # Get probabilities if available
    result_col1, result_col2 = st.columns([1, 2])
    with result_col1:
        st.success(f"**Predicted Class:**")
        st.markdown(f"## 🫘 {predicted_class}")

    with result_col2:
        bean_info = {
            "SEKER": "Small, round, white bean with smooth skin.",
            "BARBUNYA": "Medium-sized, speckled bean with reddish marks.",
            "BOMBAY": "Large-sized black bean with a distinctive appearance.",
            "CALI": "Medium to large white bean with a kidney-like shape.",
            "DERMASON": "Small to medium, white bean, most common type.",
            "HOROZ": "Medium-sized, brownish bean with a distinct pattern.",
            "SIRA": "Medium-sized, yellowish-white bean."
        }
        st.info(f"**About {predicted_class}:** {bean_info.get(predicted_class, 'Bean variety.')}")

st.divider()
st.caption("Dry Bean Classifier — Powered by SVM Pipeline | Trained on UCI Dry Bean Dataset")
