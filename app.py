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
    page_title="RideX AI",
    layout="centered"
)

# -----------------------------------
# CUSTOM CSS (CLASSY TECH UI)
# -----------------------------------
st.markdown("""
    <style>
    body {
        background-color: #0b0f1a;
        color: white;
    }

    .title {
        font-size: 60px;
        font-weight: 800;
        text-align: center;
        color: #00bfff;
        text-shadow: 0 0 10px #00bfff, 0 0 20px #0077ff, 0 0 30px #0077ff;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 22px;
        text-align: center;
        color: #ff4b4b;
        letter-spacing: 2px;
        margin-bottom: 25px;
    }

    .card {
        background-color: #111827;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 0px 25px rgba(0,255,255,0.15);
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00dbde, #fc00ff);
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE SECTION
# -----------------------------------
st.markdown('<div class="title">RideX AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Smart Road Usage Prediction System</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# IMAGE
# -----------------------------------
st.image("TVS_Hack.png", use_container_width=True)

st.markdown("---")

# -----------------------------------
# INPUT CARD
# -----------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("🚗 Vehicle Response Parameters")

st.info(
    "Enter values based on vehicle vibration, shock patterns, speed, and load behavior."
)

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

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PREDICTION
# -----------------------------------
st.markdown("---")

if st.button("🚀 Predict Road Usage"):

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

    # Confidence
    prob = model.predict_proba(input_data)
    confidence = np.max(prob) * 100

    st.markdown(f"""
        <h2 style='text-align:center; color:#00ffcc; text-shadow:0 0 10px #00ffcc;'>
        Predicted Road Type: {prediction}
        </h2>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <p style='text-align:center; color:#ffffff; font-size:18px;'>
        Confidence: {confidence:.2f}%
        </p>
    """, unsafe_allow_html=True)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.markdown("### 👨‍💻 Team Members")
st.markdown(
    """
    - **Hasitha S**  
    - **Kowshic K T**  
    - **Krishnakumar V**  
    """
)

st.caption("College of Engineering, Guindy | Intelligent Vehicle Response-Based Road Profiling")
