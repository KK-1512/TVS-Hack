import streamlit as st
import numpy as np
import joblib

# -----------------------------------
# Load trained Random Forest model
# -----------------------------------
model = joblib.load("random_forest_road_usage_model.pkl")

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Intelligent Road Usage Profiling",
    layout="centered"
)

# -----------------------------------
# App Title & Description
# -----------------------------------
st.title("Intelligent Road Usage Profiling for Two-Wheelers")

st.write(
    """
    This application predicts **road usage type** using vehicle dynamic response features 
    derived from **CAN data and minimal sensors (IMU)**.
    
    The model interprets how the **vehicle reacts to the road**, rather than measuring the road directly.
    """
)

st.markdown("---")

# -----------------------------------
# Input Section
# -----------------------------------
st.header("Enter Vehicle Response Parameters")
st.info(
        "Prediction is based on vehicle vibration severity, shock content, "
        "speed behavior, and load severity patterns."
    )

rms_acc = st.slider(
    "RMS Vertical Acceleration (m/s²)",
    min_value=0.2,
    max_value=3.5,
    value=1.0,
    step=0.1
)

kurtosis = st.slider(
    "Kurtosis (Shock Dominance)",
    min_value=2.0,
    max_value=15.0,
    value=5.0,
    step=0.1
)

shock_density = st.slider(
    "Shock Density (events per km)",
    min_value=0,
    max_value=30,
    value=5,
    step=1
)

psd_low = st.slider(
    "Low-Frequency PSD Energy",
    min_value=0.1,
    max_value=6.0,
    value=1.0,
    step=0.1
)

psd_high = st.slider(
    "High-Frequency PSD Energy",
    min_value=0.1,
    max_value=6.0,
    value=1.0,
    step=0.1
)

avg_speed = st.slider(
    "Average Vehicle Speed (km/h)",
    min_value=5.0,
    max_value=100.0,
    value=40.0,
    step=1.0
)

rlsi = st.slider(
    "Road Load Severity Index (RLSI)",
    min_value=0.2,
    max_value=12.0,
    value=3.0,
    step=0.1
)

# -----------------------------------
# Prediction Button
# -----------------------------------
st.markdown("---")

if st.button("Predict Road Usage"):
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

    st.success(f"**Predicted Road Type:** {prediction}")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption(
    "College of Engineering, Guindy | Intelligent Vehicle Response-Based Road Profiling System"
)
