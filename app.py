import streamlit as st
import numpy as np
import joblib

# -----------------------------------
# Load trained model
# -----------------------------------
model = joblib.load("random_forest_road_usage_model.pkl")

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Intelligent Road Usage Profiling",
    layout="centered"
)

# -----------------------------------
# Title
# -----------------------------------
st.title("🚀 RideX - Road Usage Prediction")

st.write(
    """
    Enter vehicle response parameters to predict **road type** 
    using ML based on CAN + IMU data.
    """
)

st.markdown("---")

# -----------------------------------
# INPUT SECTION (TYPE INPUTS)
# -----------------------------------
st.header("Enter Vehicle Parameters")

col1, col2 = st.columns(2)

with col1:
    rms_acc = st.number_input(
        "RMS Vertical Acceleration (m/s²)",
        min_value=0.2, max_value=3.5, value=1.0, step=0.1
    )

    kurtosis = st.number_input(
        "Kurtosis (Shock Dominance)",
        min_value=2.0, max_value=15.0, value=5.0, step=0.1
    )

    shock_density = st.number_input(
        "Shock Density (events/km)",
        min_value=0, max_value=30, value=5, step=1
    )

    psd_low = st.number_input(
        "Low-Frequency PSD Energy",
        min_value=0.1, max_value=6.0, value=1.0, step=0.1
    )

with col2:
    psd_high = st.number_input(
        "High-Frequency PSD Energy",
        min_value=0.1, max_value=6.0, value=1.0, step=0.1
    )

    avg_speed = st.number_input(
        "Average Vehicle Speed (km/h)",
        min_value=5.0, max_value=100.0, value=40.0, step=1.0
    )

    rlsi = st.number_input(
        "Road Load Severity Index (RLSI)",
        min_value=0.2, max_value=12.0, value=3.0, step=0.1
    )

st.markdown("---")

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("Predict Road Type"):

    input_data = np.array([[ 
        rms_acc,
        kurtosis,
        shock_density,
        psd_low,
        psd_high,
        avg_speed,
        rlsi
    ]])

    prediction = model.predict(input_data)[0]

    # Optional: Confidence score
    prob = model.predict_proba(input_data)
    confidence = np.max(prob) * 100

    st.success(f"Predicted Road Type: **{prediction}**")
    st.info(f"Confidence: {confidence:.2f}%")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.markdown("**Team Members**")
st.markdown(
    """
    - Hasitha S  
    - Kowshic K T  
    - Krishnakumar V  
    """
)

st.caption("CEG | Intelligent Vehicle Response-Based Road Profiling")
