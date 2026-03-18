import streamlit as st
import numpy as np
import joblib
import base64
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

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
# Helper: Convert image to base64
# -----------------------------------
def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -----------------------------------
# CSS STYLING
# -----------------------------------
st.markdown("""
<style>
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    position: relative;
    margin-top: 10px;
}

.logo {
    height: 70px;
}

.title-container {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
}

.title {
    font-size: 55px;
    font-weight: 800;
    color: #ff1a1a;
    margin: 0;
}

.description {
    font-size: 18px;
    line-height: 1.7;
    color: #e6e6e6;
}

.highlight {
    font-weight: 600;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.image("ceg.png", width=90)

with col2:
    st.markdown(
        "<h1 style='text-align:center; color:#ff1a1a; margin-top:10px;'>RideX</h1>",
        unsafe_allow_html=True
    )

with col3:
    st.image("tvsbn.jpg", width=90)

# -----------------------------------
# DESCRIPTION
# -----------------------------------
st.markdown("""
<div class="description">
This application predicts <span class="highlight">road usage type</span> using vehicle dynamic response features 
derived from <span class="highlight">CAN data and minimal sensors (IMU)</span>.<br><br>

The model interprets how <span class="highlight">the vehicle reacts to the road</span>, rather than measuring the road directly.
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# TIME (IST)
# -----------------------------------
# -----------------------------------
# LIVE TIME (IST)
# -----------------------------------
st_autorefresh(interval=1000, key="clock_refresh")

ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)

current_time = now.strftime("%d %B %Y | %I:%M:%S %p")

st.markdown(
    f"<h4 style='text-align:center; color:#00bfff;'>{current_time}</h4>",
    unsafe_allow_html=True
)

# -----------------------------------
# IMAGE
# -----------------------------------
st.image("TVS_Hack.png", use_container_width=True)

# -----------------------------------
# INPUT SECTION
# -----------------------------------
st.header("Vehicle Response Parameters")
st.info(
    "Prediction is based on vehicle vibration severity, shock content, "
    "speed behavior, and load severity patterns."
)

rms_acc = st.number_input("RMS Vertical Acceleration (m/s²)", 0.2, 3.5, 1.0, 0.1)
kurtosis = st.number_input("Kurtosis (Shock Dominance)", 2.0, 15.0, 5.0, 0.1)
shock_density = st.number_input("Shock Density (events per km)", 0, 30, 5, 1)
psd_low = st.number_input("Low-Frequency PSD Energy", 0.1, 6.0, 1.0, 0.1)
psd_high = st.number_input("High-Frequency PSD Energy", 0.1, 6.0, 1.0, 0.1)
avg_speed = st.number_input("Average Vehicle Speed (km/h)", 5.0, 100.0, 40.0, 1.0)
rlsi = st.number_input("Road Load Severity Index (RLSI)", 0.2, 12.0, 3.0, 0.1)

# -----------------------------------
# PREDICTION
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
# FOOTER
# -----------------------------------
st.markdown("---")

st.markdown("**Team Members**")
st.markdown("""
- **Hasitha S**  
- **Kowshic K T**  
- **Krishnakumar V**
""")

st.caption("College of Engineering, Guindy | Intelligent Vehicle Response-Based Road Profiling")
