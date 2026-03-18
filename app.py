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
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');

    .ridex-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 60px;
        font-weight: 900;
        text-align: center;
        color: #00bfff;
        text-shadow: 
            0 0 10px #00bfff,
            0 0 20px #0077ff,
            0 0 40px #0077ff;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="ridex-title">RideX</div>', unsafe_allow_html=True)

st.write(
    """
    This application predicts **road usage type** using vehicle dynamic response features 
    derived from **CAN data and minimal sensors (IMU)**.
    
    The model interprets how the **vehicle reacts to the road**, rather than measuring the road directly.
    """
)

# -----------------------------------
# Display Road Type Illustration
# -----------------------------------
st.image(
    "TVS_Hack.png",
    use_container_width=True
)

st.markdown("---")

# -----------------------------------
# Input Section (SAME AS OLD, BUT TYPING INPUT)
# -----------------------------------
st.header("Vehicle Response Parameters")
st.info(
    "Prediction is based on vehicle vibration severity, shock content, "
    "speed behavior, and load severity patterns."
)

rms_acc = st.number_input(
    "RMS Vertical Acceleration (m/s²)",
    min_value=0.2,
    max_value=3.5,
    value=1.0,
    step=0.1
)

kurtosis = st.number_input(
    "Kurtosis (Shock Dominance)",
    min_value=2.0,
    max_value=15.0,
    value=5.0,
    step=0.1
)

shock_density = st.number_input(
    "Shock Density (events per km)",
    min_value=0,
    max_value=30,
    value=5,
    step=1
)

psd_low = st.number_input(
    "Low-Frequency PSD Energy",
    min_value=0.1,
    max_value=6.0,
    value=1.0,
    step=0.1
)

psd_high = st.number_input(
    "High-Frequency PSD Energy",
    min_value=0.1,
    max_value=6.0,
    value=1.0,
    step=0.1
)

avg_speed = st.number_input(
    "Average Vehicle Speed (km/h)",
    min_value=5.0,
    max_value=100.0,
    value=40.0,
    step=1.0
)

rlsi = st.number_input(
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

st.markdown("**Team Members**")

st.markdown(
    """
    - **Hasitha S** 
    - **Kowshic K T**
    - **Krishnakumar V**
    """
)

st.caption(
    "College of Engineering, Guindy | Intelligent Vehicle Response-Based Road Profiling System"
)
