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

.header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    margin-bottom: 20px;
}

/* Logos */
.logo {
    height: 60px;
}

/* RideX Title */
.ridex-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 55px;
    font-weight: 900;
    color: #ff1a1a;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.image("ceg.png", width=80)

with col2:
    st.markdown(
        "<h1 style='text-align:center; color:red;'>RideX</h1>",
        unsafe_allow_html=True
    )

with col3:
    st.image("tvs.png", width=80)

# Description
st.markdown("""
<div class="description">
This application predicts <span class="highlight">road usage type</span> using vehicle dynamic response features 
derived from <span class="highlight">CAN data and minimal sensors (IMU)</span>.<br><br>

The model interprets how <span class="highlight">the vehicle reacts to the road</span>, rather than measuring the road directly.
</div>
""", unsafe_allow_html=True)

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
