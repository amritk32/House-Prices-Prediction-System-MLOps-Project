import streamlit as st
import requests
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Real Estate AI | Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 🌟 BACKGROUND & WHITE TEXT CSS
# ==========================================
st.markdown("""
    <style>
    /* Add a beautiful house background with a dark overlay for text readability */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), 
                    url("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?q=80&w=2075") no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Make all headings and text white with a slight shadow */
    h1, h2, h3, h4, p, label, .stMarkdown, .css-10trblm {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
        font-family: 'Inter', sans-serif;
    }

    /* Style the Sidebar to look distinct */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 20, 20, 0.85) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Highlight the metric values in green */
    [data-testid="stMetricValue"] {
        color: #4ade80 !important;
    }

    /* Input fields (white background, black text for high contrast) */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stSlider > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #000000 !important;
        font-weight: bold;
        border-radius: 6px;
    }

    /* Solid Predict Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2563eb; /* Strong Blue */
        color: white !important;
        font-weight: bold;
        font-size: 18px;
        padding: 12px;
        border: none;
        transition: 0.3s;
        box-shadow: 0px 4px 10px rgba(37, 99, 235, 0.4);
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
    
    /* Prediction Output Text */
    .final-price-text {
        font-size: 3.5rem;
        color: #4ade80 !important;
        text-align: center;
        font-weight: 800;
        text-shadow: 0px 0px 20px rgba(74, 222, 128, 0.5);
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR: MODEL METRICS
# ==========================================
st.sidebar.title("⚙️ Engine Stats")
st.sidebar.markdown("Live evaluation metrics from the training pipeline.")
st.sidebar.divider()

try:
    # Load metrics saved from main.py
    metrics = joblib.load("metrics.pkl")
    st.sidebar.markdown(f"### 🧠 Active Model")
    st.sidebar.info(f"**{metrics['model']}**")
    
    st.sidebar.metric("R² Score (Accuracy)", f"{metrics['r2']:.4f}")
    st.sidebar.metric("RMSLE (Error Rate)", f"{metrics['rmsle']:.4f}")
    
    st.sidebar.divider()
    FASTAPI_URL = "https://house-price-api-nhl3.onrender.com/predict"

except FileNotFoundError:
    st.sidebar.error("⚠️ metrics.pkl not found! Please run `python main.py` first.")

# ==========================================
# 4. MAIN UI HEADER
# ==========================================
st.title("🏡 Advanced House Price Estimator")
st.markdown("Enter the structural details below to get an AI-driven market valuation.")
st.write("---")

# ==========================================
# 5. INPUT FORM (Using Columns for layout)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=6)
    gr_liv_area = st.number_input("Above Grade Living Area (sq. ft.)", min_value=300, max_value=10000, value=1500, step=50)
    year_built = st.number_input("Construction Year", min_value=1800, max_value=2026, value=1995, step=1)

with col2:
    total_bsmt_sf = st.number_input("Basement Area (sq. ft.)", min_value=0, max_value=6000, value=1000, step=50)
    garage_cars = st.selectbox("Garage Capacity (Cars)", options=[0, 1, 2, 3, 4, 5], index=2)

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# 6. FASTAPI PREDICTION LOGIC
# ==========================================
if st.button("🚀 Calculate Market Value"):
    payload = {
        "OverallQual": int(overall_qual),
        "GrLivArea": int(gr_liv_area),
        "GarageCars": int(garage_cars),
        "TotalBsmtSF": int(total_bsmt_sf),
        "YearBuilt": int(year_built)
    }
    
    with st.spinner("Connecting to FastAPI Engine..."):
        try:
            response = requests.post(FASTAPI_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                final_price = result.get("predicted_price ") or result.get("predicted_price")
                
                st.write("---")
                st.markdown("<h3 style='text-align: center; color: #e2e8f0;'>Estimated Property Value</h3>", unsafe_allow_html=True)
                st.markdown(f"<p class='final-price-text'>${final_price:,.2f}</p>", unsafe_allow_html=True)
                st.balloons()
            else:
                st.error(f"⚠️ API Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Failed! Ensure FastAPI (`api.py`) is running on port 8000.")
