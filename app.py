"""
Smart Agricultural Analytics & Yield Prediction Dashboard
============================================================
A comprehensive, single-file Streamlit application for agricultural data
analysis, interactive data management, and machine-learning-powered
crop yield prediction.
Date: 2026-06-16
"""

# ============================================================
# SECTION 1: IMPORTS & DEPENDENCIES
# ============================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ============================================================
# SECTION 2: PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================
# Set Streamlit page configuration: title, icon, and wide layout
st.set_page_config(
    page_title="Smart Agricultural Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a modern, clean SaaS aesthetic
# We use a forest-green primary color palette with clean cards and typography
st.markdown("""
    <style>
    /* --------------------------------------------------------
       GLOBAL STYLES
       -------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Apply Inter font to all text elements */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main background: subtle off-white for readability */
    .main {
        background-color: #f8faf7 !important;
    }

    /* Remove default Streamlit padding for a tighter layout */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* --------------------------------------------------------
       SIDEBAR STYLING
       -------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #1a3c1d !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #1a3c1d !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #e8f5e9 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #c8e6c9 !important;
    }

    /* --------------------------------------------------------
       METRIC CARDS (Tab 1: Dashboard)
       -------------------------------------------------------- */
    .metric-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
        border: 1px solid #e8ede6;
        text-align: center;
        transition: box-shadow 0.2s ease;
    }
    .metric-container:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .metric-icon {
        font-size: 32px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a3c1d;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #6b7f6b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-change {
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
    }
    .metric-change.positive { color: #2e7d32; }
    .metric-change.negative { color: #c62828; }

    /* --------------------------------------------------------
       CHART CARDS
       -------------------------------------------------------- */
    .chart-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
        border: 1px solid #e8ede6;
        margin-bottom: 20px;
    }
    .chart-title {
        font-size: 15px;
        font-weight: 600;
        color: #1a3c1d;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --------------------------------------------------------
       PREDICTION ALERT CARDS (Tab 3: Predictor)
       -------------------------------------------------------- */
    .prediction-card {
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin-top: 16px;
    }
    .prediction-card.high-yield {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        color: #1b5e20;
    }
    .prediction-card.moderate-yield {
        background-color: #fff8e1;
        border: 2px solid #ffb300;
        color: #e65100;
    }
    .prediction-card.poor-yield {
        background-color: #ffebee;
        border: 2px solid #ef5350;
        color: #b71c1c;
    }
    .prediction-value {
        font-size: 48px;
        font-weight: 700;
        margin: 8px 0;
    }
    .prediction-unit {
        font-size: 16px;
        font-weight: 500;
        opacity: 0.8;
    }
    .prediction-label {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }

    /* --------------------------------------------------------
       RECOMMENDATION CARDS (Tab 4: Guide)
       -------------------------------------------------------- */
    .recommendation-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        border: 1px solid #e8ede6;
        margin-bottom: 16px;
        border-left: 5px solid #2e7d32;
    }
    .recommendation-card.corn { border-left-color: #f9a825; }
    .recommendation-card.wheat { border-left-color: #8d6e63; }
    .recommendation-card.rice { border-left-color: #00acc1; }
    .recommendation-card.soybeans { border-left-color: #7cb342; }
    .recommendation-card.cotton { border-left-color: #b0bec5; }
    .recommendation-title {
        font-size: 17px;
        font-weight: 700;
        color: #1a3c1d;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .recommendation-text {
        font-size: 14px;
        color: #4a5d4a;
        line-height: 1.7;
    }
    .condition-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }
    .condition-badge.optimal {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .condition-badge.suboptimal {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .condition-badge.poor {
        background-color: #ffebee;
        color: #c62828;
    }

    /* --------------------------------------------------------
       TABS STYLING
       -------------------------------------------------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8faf7;
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        color: #5a7a5a !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e8f5e9 !important;
        color: #1a3c1d !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a3c1d !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* --------------------------------------------------------
       BUTTON STYLES
       -------------------------------------------------------- */
    .stButton > button {
        background-color: #1a3c1d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #2e7d32 !important;
        box-shadow: 0 4px 12px rgba(26, 60, 29, 0.25) !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #e8f5e9 !important;
        color: #1a3c1d !important;
    }

    /* --------------------------------------------------------
       SLIDER STYLING
       -------------------------------------------------------- */
    [data-testid="stSlider"] > div > div > div {
        background-color: #1a3c1d !important;
    }

    /* --------------------------------------------------------
       HEADER & FOOTER
       -------------------------------------------------------- */
    .app-header {
        text-align: center;
        padding: 20px 0 30px 0;
    }
    .app-header h1 {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #1a3c1d !important;
        margin-bottom: 6px !important;
    }
    .app-header p {
        font-size: 14px !important;
        color: #6b7f6b !important;
        margin-top: 0 !important;
    }
    .app-footer {
        text-align: center;
        padding: 30px 0 10px 0;
        color: #9aaa9a;
        font-size: 12px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 3: DATA GENERATION ENGINE
# ============================================================
# This section handles the creation and loading of agricultural data.
# If crop_data.csv does not exist locally, we generate a realistic
# dataset with 500+ rows, ensuring proper correlations between
# environmental factors and crop yield.

# Define the list of supported crop types
CROP_TYPES = ['Corn', 'Wheat', 'Rice', 'Soybeans', 'Cotton']

# Define realistic agricultural parameters per crop:
# Each crop has different optimal ranges for rainfall, temperature, pH,
# N-P-K requirements, and fertilizer usage. These influence yield.
CROP_PROFILES = {
    'Corn': {
        'rainfall_mean': 750, 'rainfall_std': 150,
        'temp_mean': 24, 'temp_std': 4,
        'ph_mean': 6.2, 'ph_std': 0.5,
        'N_mean': 140, 'N_std': 30,
        'P_mean': 55, 'P_std': 15,
        'K_mean': 140, 'K_std': 25,
        'fertilizer_mean': 180, 'fertilizer_std': 40,
        'base_yield': 8.5, 'yield_sensitivity': 1.2
    },
    'Wheat': {
        'rainfall_mean': 550, 'rainfall_std': 120,
        'temp_mean': 18, 'temp_std': 4,
        'ph_mean': 6.8, 'ph_std': 0.4,
        'N_mean': 120, 'N_std': 25,
        'P_mean': 45, 'P_std': 12,
        'K_mean': 110, 'K_std': 20,
        'fertilizer_mean': 140, 'fertilizer_std': 35,
        'base_yield': 4.5, 'yield_sensitivity': 1.0
    },
    'Rice': {
        'rainfall_mean': 1400, 'rainfall_std': 250,
        'temp_mean': 28, 'temp_std': 3,
        'ph_mean': 5.8, 'ph_std': 0.5,
        'N_mean': 110, 'N_std': 25,
        'P_mean': 40, 'P_std': 12,
        'K_mean': 120, 'K_std': 20,
        'fertilizer_mean': 160, 'fertilizer_std': 40,
        'base_yield': 6.0, 'yield_sensitivity': 1.1
    },
    'Soybeans': {
        'rainfall_mean': 650, 'rainfall_std': 130,
        'temp_mean': 23, 'temp_std': 3,
        'ph_mean': 6.5, 'ph_std': 0.4,
        'N_mean': 30, 'N_std': 10,   # Soybeans fix nitrogen
        'P_mean': 50, 'P_std': 12,
        'K_mean': 100, 'K_std': 18,
        'fertilizer_mean': 80, 'fertilizer_std': 25,
        'base_yield': 3.0, 'yield_sensitivity': 0.9
    },
    'Cotton': {
        'rainfall_mean': 850, 'rainfall_std': 180,
        'temp_mean': 27, 'temp_std': 3,
        'ph_mean': 6.0, 'ph_std': 0.5,
        'N_mean': 100, 'N_std': 25,
        'P_mean': 35, 'P_std': 10,
        'K_mean': 90, 'K_std': 18,
        'fertilizer_mean': 130, 'fertilizer_std': 35,
        'base_yield': 2.5, 'yield_sensitivity': 1.0
    }
}

# CSV file path for local persistence
CSV_PATH = "crop_data.csv"


def generate_synthetic_data(n_rows=550, random_seed=42):
    """
    Generate a realistic synthetic agricultural dataset.

    Uses a composite scoring approach where each environmental factor
    contributes to an overall growth score. This creates strong, realistic
    correlations between environmental inputs and crop yield.

    Parameters:
        n_rows (int): Number of rows to generate (default 550)
        random_seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with columns:
            Crop_Type, Rainfall_mm, Temperature_C, Soil_pH,
            Nitrogen_N, Phosphorus_P, Potassium_K,
            Fertilizer_kg_ha, Yield_tons_ha
    """
    np.random.seed(random_seed)
    data_rows = []

    # Distribute rows roughly evenly across crop types
    rows_per_crop = n_rows // len(CROP_TYPES)
    remainder = n_rows % len(CROP_TYPES)

    for idx, crop in enumerate(CROP_TYPES):
        profile = CROP_PROFILES[crop]
        count = rows_per_crop + (1 if idx < remainder else 0)

        # Generate environmental variables with wide variation
        # Wider ranges ensure strong correlations with yield
        rainfall = np.random.uniform(200, 1800, count)
        temperature = np.random.uniform(12, 42, count)
        ph = np.random.uniform(4.5, 8.5, count)
        nitrogen = np.random.uniform(10, 250, count)
        phosphorus = np.random.uniform(10, 120, count)
        potassium = np.random.uniform(20, 250, count)
        fertilizer = np.random.uniform(0, 300, count)

        for i in range(count):
            # Calculate how far each factor is from crop-specific optimal
            # Score = 1 means perfect, Score = 0 means worst case
            rain_score = 1 - abs(rainfall[i] - profile['rainfall_mean']) / \
                         max(profile['rainfall_mean'] - 200, 1800 - profile['rainfall_mean'])
            temp_score = 1 - abs(temperature[i] - profile['temp_mean']) / \
                         max(profile['temp_mean'] - 12, 42 - profile['temp_mean'])
            ph_score = 1 - abs(ph[i] - profile['ph_mean']) / \
                       max(profile['ph_mean'] - 4.5, 8.5 - profile['ph_mean'])

            # Nutrient scores: more is better up to a point (diminishing returns)
            n_score = min(1, nitrogen[i] / (profile['N_mean'] * 1.5))
            p_score = min(1, phosphorus[i] / (profile['P_mean'] * 1.5))
            k_score = min(1, potassium[i] / (profile['K_mean'] * 1.5))
            fert_score = min(1, fertilizer[i] / (profile['fertilizer_mean'] * 1.5))

            # Weighted composite: rainfall and temperature dominate real agriculture
            composite = (0.25 * rain_score +
                        0.25 * temp_score +
                        0.10 * ph_score +
                        0.10 * n_score +
                        0.10 * p_score +
                        0.10 * k_score +
                        0.10 * fert_score)

            # Yield = base yield scaled by composite score + small noise
            yield_val = profile['base_yield'] * composite * np.random.normal(1, 0.05)
            yield_val = max(0.3, round(yield_val, 2))

            data_rows.append({
                'Crop_Type': crop,
                'Rainfall_mm': round(rainfall[i], 1),
                'Temperature_C': round(temperature[i], 1),
                'Soil_pH': round(ph[i], 2),
                'Nitrogen_N': round(nitrogen[i], 1),
                'Phosphorus_P': round(phosphorus[i], 1),
                'Potassium_K': round(potassium[i], 1),
                'Fertilizer_kg_ha': round(fertilizer[i], 1),
                'Yield_tons_ha': yield_val
            })

    df = pd.DataFrame(data_rows)
    # Shuffle the rows for randomness
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df


def load_or_generate_data():
    """
    Load the dataset from crop_data.csv if it exists.
    Otherwise, generate synthetic data and save it locally.

    Returns:
        pd.DataFrame: The active dataset
    """
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # Ensure all expected columns are present
            expected_cols = ['Crop_Type', 'Rainfall_mm', 'Temperature_C', 'Soil_pH',
                           'Nitrogen_N', 'Phosphorus_P', 'Potassium_K',
                           'Fertilizer_kg_ha', 'Yield_tons_ha']
            if all(col in df.columns for col in expected_cols):
                return df
            else:
                st.warning("CSV file missing expected columns. Regenerating data...")
        except Exception as e:
            st.warning(f"Error reading CSV: {e}. Regenerating data...")

    # Generate new data if file doesn't exist or is invalid
    df = generate_synthetic_data()
    df.to_csv(CSV_PATH, index=False)
    return df


# ============================================================
# SECTION 4: MACHINE LEARNING ENGINE
# ============================================================
def train_yield_predictor(df):
    """
    Train a Random Forest Regressor to predict crop yield based on
    environmental factors and crop type.

    Parameters:
        df (pd.DataFrame): The training dataset

    Returns:
        tuple: (trained_model, label_encoder, performance_metrics_dict)
    """
    try:
        # Create a copy to avoid modifying the original DataFrame
        data = df.copy()

        # Encode the Crop_Type categorical variable into numeric labels
        le = LabelEncoder()
        data['Crop_Type_Encoded'] = le.fit_transform(data['Crop_Type'])

        # Define feature columns (all inputs except the target)
        feature_cols = ['Crop_Type_Encoded', 'Rainfall_mm', 'Temperature_C',
                        'Soil_pH', 'Nitrogen_N', 'Phosphorus_P',
                        'Potassium_K', 'Fertilizer_kg_ha']
        X = data[feature_cols]
        y = data['Yield_tons_ha']

        # Split data for evaluation (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the Random Forest model
        # 150 trees provides good accuracy without excessive training time
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = model.predict(X_test)
        metrics = {
            'r2_score': round(r2_score(y_test, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        }

        return model, le, metrics

    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None


def predict_yield(model, label_encoder, crop, rainfall, temperature, ph, nitrogen,
                  phosphorus, potassium, fertilizer):
    """
    Predict crop yield given input parameters using the trained model.

    Parameters:
        model: Trained RandomForestRegressor
        label_encoder: Fitted LabelEncoder for Crop_Type
        crop (str): Crop type name
        rainfall, temperature, ph, nitrogen, phosphorus, potassium, fertilizer: float values

    Returns:
        float: Predicted yield in tons per hectare
    """
    try:
        # Convert crop name to numeric label
        crop_encoded = label_encoder.transform([crop])[0]

        # Build a DataFrame with the same column names used during training
        # This avoids sklearn warnings about feature names
        features = pd.DataFrame([{
            'Crop_Type_Encoded': crop_encoded,
            'Rainfall_mm': rainfall,
            'Temperature_C': temperature,
            'Soil_pH': ph,
            'Nitrogen_N': nitrogen,
            'Phosphorus_P': phosphorus,
            'Potassium_K': potassium,
            'Fertilizer_kg_ha': fertilizer
        }])

        prediction = model.predict(features)[0]
        return max(0, round(prediction, 2))
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# ============================================================
# SECTION 5: SESSION STATE MANAGEMENT
# ============================================================
# Streamlit's session_state persists data across user interactions.
# We use it to store the active dataset, trained model, and UI state.

def initialize_session_state():
    """
    Initialize session state variables if they don't already exist.
    This ensures data and models persist across widget interactions.
    """
    if 'df' not in st.session_state:
        st.session_state.df = load_or_generate_data()

    if 'model' not in st.session_state or 'label_encoder' not in st.session_state:
        model, le, metrics = train_yield_predictor(st.session_state.df)
        st.session_state.model = model
        st.session_state.label_encoder = le
        st.session_state.model_metrics = metrics

    if 'prediction_inputs' not in st.session_state:
        # Default slider values for the predictor tab
        st.session_state.prediction_inputs = {
            'crop': 'Corn',
            'rainfall': 750.0,
            'temperature': 24.0,
            'ph': 6.5,
            'nitrogen': 130.0,
            'phosphorus': 50.0,
            'potassium': 120.0,
            'fertilizer': 150.0
        }


# ============================================================
# SECTION 6: SIDEBAR (Global Controls)
# ============================================================
def render_sidebar():
    """
    Render the left sidebar with global application controls,
    branding, and summary statistics.
    """
    with st.sidebar:
        # ---- Branding Header ----
        st.markdown("""
            <div style="text-align: center; padding-bottom: 20px;">
                <div style="font-size: 48px;">🌾</div>
                <div style="font-size: 20px; font-weight: 700; color: #e8f5e9;">
                    AgriAnalytics
                </div>
                <div style="font-size: 12px; color: #a5d6a7; margin-top: 4px;">
                    Smart Yield Intelligence
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ---- Global Filters ----
        st.markdown("##### 🌍 Global Filters")

        # Crop filter multi-select
        selected_crops = st.multiselect(
            "Filter Crop Types",
            options=CROP_TYPES,
            default=CROP_TYPES,
            help="Select which crops to include in dashboard analytics"
        )

        # Rainfall range filter
        rain_min, rain_max = st.slider(
            "Rainfall Range (mm)",
            min_value=200, max_value=2000,
            value=(200, 2000),
            help="Filter data by rainfall range"
        )

        # Temperature range filter
        temp_min, temp_max = st.slider(
            "Temperature Range (°C)",
            min_value=10, max_value=45,
            value=(10, 45),
            help="Filter data by temperature range"
        )

        # ---- Data Actions ----
        st.markdown("---")
        st.markdown("##### ⚡ Quick Actions")

        # Button to regenerate synthetic data
        if st.button("🔄 Regenerate Synthetic Data", use_container_width=True):
            try:
                st.session_state.df = generate_synthetic_data()
                st.session_state.df.to_csv(CSV_PATH, index=False)
                model, le, metrics = train_yield_predictor(st.session_state.df)
                st.session_state.model = model
                st.session_state.label_encoder = le
                st.session_state.model_metrics = metrics
                st.success("Data regenerated and model retrained!")
                st.rerun()
            except Exception as e:
                st.error(f"Error regenerating data: {e}")

        # Button to download current data
        csv_buffer = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download Current CSV",
            data=csv_buffer,
            file_name="crop_data_export.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ---- System Info ----
        st.markdown("---")
        st.markdown("##### 📈 System Status")
        df = st.session_state.df
        st.markdown(f"""
            <div style="font-size: 12px; color: #a5d6a7;">
                <div>📝 Records: <b>{len(df)}</b></div>
                <div>🌱 Crops: <b>{df['Crop_Type'].nunique()}</b></div>
                <div>🤖 Model: <b>Random Forest</b></div>
            </div>
        """, unsafe_allow_html=True)

        # Show model metrics if available
        if 'model_metrics' in st.session_state and st.session_state.model_metrics:
            m = st.session_state.model_metrics
            st.markdown(f"""
                <div style="font-size: 11px; color: #81c784; margin-top: 8px;">
                    <div>R² Score: <b>{m['r2_score']}</b></div>
                    <div>RMSE: <b>{m['rmse']}</b> t/ha</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
            <div style="font-size: 11px; color: #81c784; text-align: center;">
                AgriAnalytics v1.0<br>
                Built with Streamlit + Plotly
            </div>
        """, unsafe_allow_html=True)

    # Return filter values for use in dashboard
    return selected_crops, rain_min, rain_max, temp_min, temp_max


# ============================================================
# SECTION 7: TAB 1 — REAL-TIME ANALYTICS DASHBOARD
# ============================================================
def render_analytics_dashboard(df, selected_crops, rain_min, rain_max, temp_min, temp_max):
    """
    Render the main analytics dashboard with metric cards and
    interactive Plotly visualizations.

    Parameters:
        df (pd.DataFrame): The full dataset
        selected_crops (list): Crops selected in sidebar filter
        rain_min, rain_max (int): Rainfall filter bounds
        temp_min, temp_max (int): Temperature filter bounds
    """
    # ---- Apply global filters ----
    filtered_df = df[
        (df['Crop_Type'].isin(selected_crops)) &
        (df['Rainfall_mm'] >= rain_min) & (df['Rainfall_mm'] <= rain_max) &
        (df['Temperature_C'] >= temp_min) & (df['Temperature_C'] <= temp_max)
    ].copy()

    # Handle empty filtered state
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your filter criteria in the sidebar.")
        return

    # ---- App Header ----
    st.markdown("""
        <div class="app-header">
            <h1>🌾 Smart Agricultural Analytics</h1>
            <p>Real-time insights into crop performance, environmental correlations, and yield predictions</p>
        </div>
    """, unsafe_allow_html=True)

    # ---- Metric Cards Row ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_crops = len(filtered_df)
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-icon">🌱</div>
                <div class="metric-value">{total_crops:,}</div>
                <div class="metric-label">Crop Records</div>
                <div class="metric-change positive">+{total_crops - len(df) + len(filtered_df)} active</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_yield = filtered_df['Yield_tons_ha'].mean()
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-icon">📈</div>
                <div class="metric-value">{avg_yield:.2f}</div>
                <div class="metric-label">Avg Yield (t/ha)</div>
                <div class="metric-change positive">Global average</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        optimal_rain = filtered_df.loc[filtered_df['Yield_tons_ha'].idxmax(), 'Rainfall_mm']
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-icon">🌧️</div>
                <div class="metric-value">{optimal_rain:.0f}</div>
                <div class="metric-label">Optimal Rainfall (mm)</div>
                <div class="metric-change positive">Peak performance</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        top_crop = filtered_df.groupby('Crop_Type')['Yield_tons_ha'].mean().idxmax()
        top_yield = filtered_df.groupby('Crop_Type')['Yield_tons_ha'].mean().max()
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-icon">🏆</div>
                <div class="metric-value">{top_crop}</div>
                <div class="metric-label">Top Crop</div>
                <div class="metric-change positive">{top_yield:.2f} t/ha avg</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Row 1: Bar Chart + Scatter Plot ----
    col_left, col_right = st.columns(2)

    with col_left:
        # Average Yield by Crop Type — grouped bar chart
        avg_by_crop = filtered_df.groupby('Crop_Type')['Yield_tons_ha'].mean().reset_index()
        avg_by_crop = avg_by_crop.sort_values('Yield_tons_ha', ascending=True)

        fig_bar = px.bar(
            avg_by_crop,
            x='Yield_tons_ha',
            y='Crop_Type',
            orientation='h',
            title=None,
            labels={'Yield_tons_ha': 'Avg Yield (t/ha)', 'Crop_Type': ''},
            color='Yield_tons_ha',
            color_continuous_scale='Greens',
            template='plotly_white',
            height=360
        )
        fig_bar.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', size=12),
            title=dict(text='<b>Average Yield by Crop Type</b>', x=0, font=dict(size=14, color='#1a3c1d'))
        )
        fig_bar.update_traces(marker_line_color='#1a3c1d', marker_line_width=0.5)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Scatter: Rainfall vs Yield (color by Crop Type)
        fig_scatter = px.scatter(
            filtered_df,
            x='Rainfall_mm',
            y='Yield_tons_ha',
            color='Crop_Type',
            labels={'Rainfall_mm': 'Rainfall (mm)', 'Yield_tons_ha': 'Yield (t/ha)', 'Crop_Type': 'Crop'},
            template='plotly_white',
            height=360,
            opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_scatter.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=10)),
            title=dict(text='<b>Rainfall vs Crop Yield</b>', x=0, font=dict(size=14, color='#1a3c1d'))
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Row 2: Distribution Plot + Temperature Scatter ----
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        # Soil pH Distribution — overlaid histogram
        fig_ph = px.histogram(
            filtered_df,
            x='Soil_pH',
            color='Crop_Type',
            nbins=30,
            barmode='overlay',
            labels={'Soil_pH': 'Soil pH Level', 'count': 'Frequency', 'Crop_Type': 'Crop'},
            template='plotly_white',
            height=360,
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.6
        )
        fig_ph.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=10)),
            title=dict(text='<b>Soil pH Distribution by Crop</b>', x=0, font=dict(size=14, color='#1a3c1d'))
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_ph, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right2:
        # Scatter: Temperature vs Yield
        fig_temp = px.scatter(
            filtered_df,
            x='Temperature_C',
            y='Yield_tons_ha',
            color='Crop_Type',
            labels={'Temperature_C': 'Temperature (°C)', 'Yield_tons_ha': 'Yield (t/ha)', 'Crop_Type': 'Crop'},
            template='plotly_white',
            height=360,
            opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_temp.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=10)),
            title=dict(text='<b>Temperature vs Crop Yield</b>', x=0, font=dict(size=14, color='#1a3c1d'))
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Summary Statistics Table ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📋 Crop Performance Summary Statistics</div>', unsafe_allow_html=True)

    summary_stats = filtered_df.groupby('Crop_Type').agg({
        'Yield_tons_ha': ['mean', 'std', 'min', 'max'],
        'Rainfall_mm': 'mean',
        'Temperature_C': 'mean',
        'Soil_pH': 'mean',
        'Fertilizer_kg_ha': 'mean'
    }).round(2)
    summary_stats.columns = ['Avg Yield', 'Yield Std', 'Min Yield', 'Max Yield',
                             'Avg Rainfall', 'Avg Temp', 'Avg pH', 'Avg Fertilizer']
    summary_stats = summary_stats.reset_index()
    st.dataframe(summary_stats, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 8: TAB 2 — LIVE DATA MANAGEMENT (Editable Ledger)
# ============================================================
def render_data_management():
    """
    Render the interactive data management tab where users can:
    - Edit cell values directly in a data grid
    - Add new rows
    - Delete rows
    - Commit changes to CSV and retrain the model
    - Upload a new CSV file to replace the dataset
    """
    st.markdown("""
        <div class="app-header">
            <h1>🚜 Live Data Management</h1>
            <p>Edit, add, or delete crop records. Commit changes to update the ML model.</p>
        </div>
    """, unsafe_allow_html=True)

    # ---- File Uploader Section ----
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📁 Upload New Dataset (Optional)</div>', unsafe_allow_html=True)
    st.info("Upload a CSV file to replace the current dataset. The file must contain columns: Crop_Type, Rainfall_mm, Temperature_C, Soil_pH, Nitrogen_N, Phosphorus_P, Potassium_K, Fertilizer_kg_ha, Yield_tons_ha")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="data_upload")
    if uploaded_file is not None:
        try:
            new_df = pd.read_csv(uploaded_file)
            expected_cols = ['Crop_Type', 'Rainfall_mm', 'Temperature_C', 'Soil_pH',
                           'Nitrogen_N', 'Phosphorus_P', 'Potassium_K',
                           'Fertilizer_kg_ha', 'Yield_tons_ha']
            if all(col in new_df.columns for col in expected_cols):
                st.session_state.df = new_df
                st.session_state.df.to_csv(CSV_PATH, index=False)
                model, le, metrics = train_yield_predictor(st.session_state.df)
                st.session_state.model = model
                st.session_state.label_encoder = le
                st.session_state.model_metrics = metrics
                st.success(f"✅ Uploaded {len(new_df)} records. Model retrained successfully!")
                st.rerun()
            else:
                missing = [c for c in expected_cols if c not in new_df.columns]
                st.error(f"Missing required columns: {missing}")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Interactive Data Editor ----
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">✏️ Editable Data Ledger</div>', unsafe_allow_html=True)
    st.caption("Click on any cell to edit its value. Use the rightmost column to delete rows. Add new rows at the bottom.")

    df = st.session_state.df.copy()

    # Configure column types for the data editor
    column_config = {
        'Crop_Type': st.column_config.SelectboxColumn(
            'Crop Type',
            options=CROP_TYPES,
            required=True
        ),
        'Rainfall_mm': st.column_config.NumberColumn(
            'Rainfall (mm)',
            min_value=0, max_value=3000,
            step=10
        ),
        'Temperature_C': st.column_config.NumberColumn(
            'Temperature (°C)',
            min_value=-10, max_value=55,
            step=0.5
        ),
        'Soil_pH': st.column_config.NumberColumn(
            'Soil pH',
            min_value=3.0, max_value=10.0,
            step=0.1
        ),
        'Nitrogen_N': st.column_config.NumberColumn(
            'Nitrogen (N)',
            min_value=0, max_value=300,
            step=5
        ),
        'Phosphorus_P': st.column_config.NumberColumn(
            'Phosphorus (P)',
            min_value=0, max_value=150,
            step=5
        ),
        'Potassium_K': st.column_config.NumberColumn(
            'Potassium (K)',
            min_value=0, max_value=300,
            step=5
        ),
        'Fertilizer_kg_ha': st.column_config.NumberColumn(
            'Fertilizer (kg/ha)',
            min_value=0, max_value=500,
            step=10
        ),
        'Yield_tons_ha': st.column_config.NumberColumn(
            'Yield (t/ha)',
            min_value=0, max_value=50,
            step=0.1
        )
    }

    # Use Streamlit's data_editor for in-place editing
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        num_rows="dynamic",   # Allows adding new rows at the bottom
        use_container_width=True,
        height=500,
        key="crop_data_editor"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Commit Changes Button ----
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("💾 Commit & Save Changes", type="primary", use_container_width=True):
            try:
                with st.spinner("Saving data and retraining model..."):
                    # Validate the edited data
                    edited_df = edited_df.dropna(subset=['Crop_Type'])
                    if len(edited_df) < 10:
                        st.error("Dataset must have at least 10 rows for the model to train properly.")
                        return

                    # Save to CSV
                    edited_df.to_csv(CSV_PATH, index=False)
                    st.session_state.df = edited_df

                    # Retrain the ML model with updated data
                    model, le, metrics = train_yield_predictor(st.session_state.df)
                    st.session_state.model = model
                    st.session_state.label_encoder = le
                    st.session_state.model_metrics = metrics

                st.success(f"✅ Saved {len(edited_df)} rows. Model retrained! (R² = {metrics['r2_score']})")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"Error saving changes: {e}")

    with col_btn2:
        st.caption("Clicking 'Commit & Save Changes' will overwrite crop_data.csv and retrain the Random Forest model automatically.")


# ============================================================
# SECTION 9: TAB 3 — SMART MULTI-CROP YIELD PREDICTOR
# ============================================================
def render_yield_predictor():
    """
    Render the interactive yield prediction tab with slider controls
    for environmental factors and a color-coded prediction result card.
    """
    st.markdown("""
        <div class="app-header">
            <h1>🔮 Smart Multi-Crop Yield Predictor</h1>
            <p>Adjust environmental factors to simulate and predict crop yield in real-time</p>
        </div>
    """, unsafe_allow_html=True)

    # ---- Two-column layout: Controls | Prediction ----
    col_controls, col_prediction = st.columns([1.2, 1])

    with col_controls:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🎛️ Environmental Controls</div>', unsafe_allow_html=True)

        # Crop selection dropdown
        selected_crop = st.selectbox(
            "Select Crop Type",
            options=CROP_TYPES,
            index=CROP_TYPES.index(st.session_state.prediction_inputs['crop']),
            help="Choose which crop to simulate"
        )

        # Slider controls for all environmental factors
        # We use realistic ranges for each parameter
        rainfall = st.slider(
            "🌧️ Rainfall (mm)",
            min_value=200, max_value=2000,
            value=int(st.session_state.prediction_inputs['rainfall']),
            step=10,
            help="Annual rainfall in millimeters"
        )

        temperature = st.slider(
            "🌡️ Temperature (°C)",
            min_value=10, max_value=45,
            value=int(st.session_state.prediction_inputs['temperature']),
            step=1,
            help="Average growing temperature in Celsius"
        )

        ph = st.slider(
            "⚗️ Soil pH",
            min_value=4.5, max_value=8.5,
            value=float(st.session_state.prediction_inputs['ph']),
            step=0.1,
            help="Soil acidity/alkalinity level"
        )

        fertilizer = st.slider(
            "🧪 Fertilizer Input (kg/ha)",
            min_value=0, max_value=300,
            value=int(st.session_state.prediction_inputs['fertilizer']),
            step=5,
            help="Total fertilizer application rate"
        )

        # Nitrogen slider (context-aware based on crop)
        nitrogen = st.slider(
            "🔬 Nitrogen - N (kg/ha)",
            min_value=5, max_value=250,
            value=int(st.session_state.prediction_inputs['nitrogen']),
            step=5,
            help="Nitrogen content in soil"
        )

        phosphorus = st.slider(
            "🔬 Phosphorus - P (kg/ha)",
            min_value=5, max_value=150,
            value=int(st.session_state.prediction_inputs['phosphorus']),
            step=5,
            help="Phosphorus content in soil"
        )

        potassium = st.slider(
            "🔬 Potassium - K (kg/ha)",
            min_value=10, max_value=250,
            value=int(st.session_state.prediction_inputs['potassium']),
            step=5,
            help="Potassium content in soil"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col_prediction:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        # Retrieve the trained model and label encoder from session state
        model = st.session_state.get('model')
        le = st.session_state.get('label_encoder')

        if model is not None and le is not None:
            # Run the prediction with current slider values
            prediction = predict_yield(
                model, le, selected_crop,
                rainfall, temperature, ph,
                nitrogen, phosphorus, potassium, fertilizer
            )

            if prediction is not None:
                # Determine yield category for color-coding
                # Thresholds are relative to crop's base yield
                base = CROP_PROFILES[selected_crop]['base_yield']
                if prediction >= base * 1.1:
                    card_class = "high-yield"
                    label = "🟢 Excellent Yield"
                    advice = "Conditions are highly favorable. Expect strong harvest performance."
                elif prediction >= base * 0.85:
                    card_class = "moderate-yield"
                    label = "🟡 Moderate Yield"
                    advice = "Yield is acceptable. Consider optimizing fertilizer or irrigation."
                else:
                    card_class = "poor-yield"
                    label = "🔴 Suboptimal Yield"
                    advice = "Conditions are challenging. Review environmental factors for improvements."

                # Render the prediction card with color coding
                st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <div class="prediction-label">{label}</div>
                        <div class="prediction-value">{prediction}</div>
                        <div class="prediction-unit">tons per hectare</div>
                        <div style="margin-top: 16px; font-size: 13px; line-height: 1.5;">
                            {advice}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Feature importance visualization
                st.markdown("<br>", unsafe_allow_html=True)

                # Get feature importances from the trained Random Forest
                feature_names = ['Crop', 'Rainfall', 'Temp', 'pH', 'N', 'P', 'K', 'Fertilizer']
                importances = model.feature_importances_

                imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

                fig_imp = px.bar(
                    imp_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Greens',
                    template='plotly_white', height=280
                )
                fig_imp.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter, sans-serif', size=11),
                    title=dict(text='<b>Feature Importance</b>', x=0, font=dict(size=12, color='#1a3c1d'))
                )
                st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

                # Model performance info
                if 'model_metrics' in st.session_state and st.session_state.model_metrics:
                    m = st.session_state.model_metrics
                    st.caption(f"Model Performance: R² = {m['r2_score']} | RMSE = {m['rmse']} t/ha")
            else:
                st.error("Unable to generate prediction. Please check your inputs.")
        else:
            st.warning("⚠️ Model not trained yet. Please ensure data is loaded.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Save current inputs to session state for persistence ----
    st.session_state.prediction_inputs = {
        'crop': selected_crop,
        'rainfall': rainfall,
        'temperature': temperature,
        'ph': ph,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium,
        'fertilizer': fertilizer
    }

    # ---- Historical Comparison Section ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📈 Your Prediction vs. Historical Data</div>', unsafe_allow_html=True)

    # Compare the current prediction against historical data for the selected crop
    crop_history = st.session_state.df[st.session_state.df['Crop_Type'] == selected_crop]
    if len(crop_history) > 0 and prediction is not None:
        hist_mean = crop_history['Yield_tons_ha'].mean()
        hist_std = crop_history['Yield_tons_ha'].std()

        fig_compare = go.Figure()

        # Historical range (mean ± std)
        fig_compare.add_trace(go.Bar(
            x=['Historical Avg', 'Your Prediction'],
            y=[hist_mean, prediction],
            marker_color=['#a5d6a7', '#1a3c1d'],
            text=[f'{hist_mean:.2f}', f'{prediction:.2f}'],
            textposition='outside',
            textfont=dict(size=14, family='Inter', color='#1a3c1d'),
            width=0.5
        ))

        # Add error bars for historical std
        fig_compare.update_traces(
            error_y=dict(type='data', array=[hist_std, 0], visible=True,
                        color='#81c784', thickness=1.5)
        )

        fig_compare.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', size=12),
            showlegend=False,
            yaxis_title='Yield (t/ha)',
            title=dict(text=f'<b>{selected_crop}: Historical vs. Predicted</b>',
                      x=0, font=dict(size=13, color='#1a3c1d'))
        )

        st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})

        # Contextual insight
        if prediction > hist_mean + hist_std:
            st.success(f"🌟 Your predicted yield ({prediction} t/ha) is significantly above the historical average ({hist_mean:.2f} t/ha) for {selected_crop}.")
        elif prediction > hist_mean:
            st.info(f"👍 Your predicted yield ({prediction} t/ha) is above the historical average ({hist_mean:.2f} t/ha) for {selected_crop}.")
        elif prediction > hist_mean - hist_std:
            st.warning(f"⚠️ Your predicted yield ({prediction} t/ha) is below the historical average ({hist_mean:.2f} t/ha) for {selected_crop}.")
        else:
            st.error(f"🔴 Your predicted yield ({prediction} t/ha) is significantly below the historical average ({hist_mean:.2f} t/ha) for {selected_crop}. Consider adjusting inputs.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 10: TAB 4 — CROP RECOMMENDATION GUIDE
# ============================================================
def render_crop_guide():
    """
    Render a static/rule-based crop recommendation guide that provides
    farming advice based on the user's current slider inputs from the
    prediction tab. This creates a bridge between the prediction tool
    and actionable agricultural guidance.
    """
    st.markdown("""
        <div class="app-header">
            <h1>🌿 Crop Recommendation Guide</h1>
            <p>Expert guidance for optimal growing conditions based on your environmental inputs</p>
        </div>
    """, unsafe_allow_html=True)

    # Retrieve the current prediction inputs to contextualize recommendations
    inputs = st.session_state.get('prediction_inputs', {})
    current_crop = inputs.get('crop', 'Corn')
    current_rain = inputs.get('rainfall', 750)
    current_temp = inputs.get('temperature', 24)
    current_ph = inputs.get('ph', 6.5)
    current_fert = inputs.get('fertilizer', 150)

    # ---- Your Current Conditions Summary ----
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🎯 Your Current Simulated Conditions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Crop", current_crop)
    with c2:
        st.metric("Rainfall", f"{current_rain:.0f} mm")
    with c3:
        st.metric("Temperature", f"{current_temp:.0f}°C")
    with c4:
        st.metric("Soil pH", f"{current_ph:.1f}")
    with c5:
        st.metric("Fertilizer", f"{current_fert:.0f} kg/ha")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Helper: Determine condition badge class ----
    def condition_badge(value, optimal_min, optimal_max, label):
        """Return an HTML badge indicating if a condition is optimal, suboptimal, or poor."""
        tolerance = (optimal_max - optimal_min) * 0.25
        if optimal_min - tolerance <= value <= optimal_max + tolerance:
            return f'<span class="condition-badge optimal">✓ {label}: Optimal</span>'
        elif optimal_min - tolerance * 2 <= value <= optimal_max + tolerance * 2:
            return f'<span class="condition-badge suboptimal">~ {label}: Suboptimal</span>'
        else:
            return f'<span class="condition-badge poor">✗ {label}: Poor</span>'

    # ---- Individual Crop Recommendation Cards ----
    for crop in CROP_TYPES:
        profile = CROP_PROFILES[crop]

        # Calculate optimal ranges from the profile means
        rain_opt_min = profile['rainfall_mean'] - profile['rainfall_std']
        rain_opt_max = profile['rainfall_mean'] + profile['rainfall_std']
        temp_opt_min = profile['temp_mean'] - profile['temp_std']
        temp_opt_max = profile['temp_mean'] + profile['temp_std']
        ph_opt_min = profile['ph_mean'] - profile['ph_std']
        ph_opt_max = profile['ph_mean'] + profile['ph_std']
        fert_opt_min = profile['fertilizer_mean'] - profile['fertilizer_std']
        fert_opt_max = profile['fertilizer_mean'] + profile['fertilizer_std']

        # Build condition badges based on user's current inputs
        rain_badge = condition_badge(current_rain, rain_opt_min, rain_opt_max, "Rainfall")
        temp_badge = condition_badge(current_temp, temp_opt_min, temp_opt_max, "Temperature")
        ph_badge = condition_badge(current_ph, ph_opt_min, ph_opt_max, "pH")
        fert_badge = condition_badge(current_fert, fert_opt_min, fert_opt_max, "Fertilizer")

        # Build crop-specific advice text
        advice_map = {
            'Corn': (
                "Corn thrives in warm climates with moderate rainfall. It is a heavy feeder of nitrogen "
                "and requires well-drained, slightly acidic to neutral soil. Ensure adequate spacing "
                "between rows for proper pollination. Monitor for pests like corn borers and aphids. "
                "Best planted in spring when soil temperatures consistently exceed 10°C."
            ),
            'Wheat': (
                "Wheat prefers cooler temperatures and moderate rainfall. It grows best in loamy soils "
                "with good drainage. Wheat requires a dormant cold period (vernalization) for optimal "
                "heading. Apply nitrogen in split doses—at sowing and during stem elongation. "
                "Watch for rust diseases and powdery mildew in humid conditions."
            ),
            'Rice': (
                "Rice is a water-intensive crop that thrives in flooded paddy conditions with high "
                "temperatures and humidity. It prefers slightly acidic soils. Ensure reliable irrigation "
                "or high rainfall during the growing season. Transplant seedlings at 3-4 weeks for "
                "best results. Control weeds early and watch for blast disease and brown plant hoppers."
            ),
            'Soybeans': (
                "Soybeans are nitrogen-fixing legumes that require less nitrogen fertilizer than other crops. "
                "They prefer warm temperatures and well-drained loamy soils. Inoculate seeds with "
                "rhizobium bacteria for better nitrogen fixation. Soybeans are sensitive to waterlogging "
                "but need consistent moisture during pod filling. Rotate with cereals to break disease cycles."
            ),
            'Cotton': (
                "Cotton requires a long, warm growing season with plenty of sunshine. It is drought-tolerant "
                "but benefits from timely irrigation during flowering and boll development. Cotton prefers "
                "deep, well-drained soils. Monitor closely for bollworm and whitefly infestations. "
                "Harvest promptly when bolls open to preserve fiber quality."
            )
        }

        crop_advice = advice_map.get(crop, "No specific advice available.")

        # Render the recommendation card
        emoji_map = {'Corn': '🌽', 'Wheat': '🌾', 'Rice': '🍚', 'Soybeans': '🫘', 'Cotton': '☁️'}

        st.markdown(f"""
            <div class="recommendation-card {crop.lower()}">
                <div class="recommendation-title">{emoji_map.get(crop, '🌱')} {crop}</div>
                <div style="margin-bottom: 12px;">
                    {rain_badge} {temp_badge} {ph_badge} {fert_badge}
                </div>
                <div class="recommendation-text">{crop_advice}</div>
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #e8ede6;">
                    <span style="font-size: 12px; font-weight: 600; color: #5a7a5a;">
                        Optimal Range — Rainfall: {rain_opt_min:.0f}-{rain_opt_max:.0f} mm |
                        Temp: {temp_opt_min:.0f}-{temp_opt_max:.0f}°C |
                        pH: {ph_opt_min:.1f}-{ph_opt_max:.1f}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ---- General Farming Tips Section ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">💡 Universal Best Practices for All Crops</div>', unsafe_allow_html=True)

    tips = [
        ("🧪 Soil Testing", "Test your soil annually to understand nutrient levels, pH, and organic matter content. This guides precise fertilizer application."),
        ("💧 Water Management", "Implement drip irrigation or sprinkler systems to reduce water waste. Time irrigation for early morning or late evening."),
        ("🌱 Crop Rotation", "Rotate crops each season to prevent soil nutrient depletion and break pest/disease cycles. Avoid planting the same family consecutively."),
        ("🐛 Integrated Pest Management (IPM)", "Use a combination of biological, cultural, and chemical methods to control pests sustainably and minimize pesticide resistance."),
        ("📊 Record Keeping", "Maintain detailed logs of planting dates, inputs, weather, and yields. This data is invaluable for optimizing future seasons."),
        ("🌿 Cover Crops", "Plant cover crops like clover or rye during off-seasons to prevent erosion, fix nitrogen, and improve soil structure."),
        ("☀️ Weather Monitoring", "Use local weather stations or apps to track rainfall, temperature, and frost warnings for timely decision-making."),
        ("🔄 Precision Agriculture", "Consider GPS-guided equipment and variable-rate technology to apply inputs only where needed, reducing costs and environmental impact.")
    ]

    for title, content in tips:
        st.markdown(f"""
            <div style="padding: 10px 0; border-bottom: 1px solid #f0f0f0;">
                <span style="font-weight: 600; color: #1a3c1d; font-size: 13px;">{title}</span>
                <p style="margin: 4px 0 0 0; color: #5a7a5a; font-size: 12px; line-height: 1.6;">{content}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# SECTION 11: MAIN APPLICATION ENTRY POINT
# ============================================================
def main():
    """
    Main application entry point.
    Initializes session state, renders the sidebar, and manages
    navigation between the four dashboard tabs.
    """
    # Initialize session state (data + model)
    initialize_session_state()

    # Render sidebar and capture filter values
    selected_crops, rain_min, rain_max, temp_min, temp_max = render_sidebar()

    # ---- Main Content: Four Tabs ----
    # st.tabs creates a horizontal navigation bar at the top of the main area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-Time Analytics",
        "🚜 Live Data Management",
        "🔮 Smart Yield Predictor",
        "🌿 Crop Recommendation Guide"
    ])

    # ---- Tab 1: Analytics Dashboard ----
    with tab1:
        render_analytics_dashboard(
            st.session_state.df, selected_crops,
            rain_min, rain_max, temp_min, temp_max
        )

    # ---- Tab 2: Data Management ----
    with tab2:
        render_data_management()

    # ---- Tab 3: Yield Predictor ----
    with tab3:
        render_yield_predictor()

    # ---- Tab 4: Crop Guide ----
    with tab4:
        render_crop_guide()

    # ---- Footer ----
    st.markdown("""
        <div class="app-footer">
            AgriAnalytics Dashboard v1.0 | Built with Python, Streamlit & Plotly<br>
            Data is simulated for demonstration purposes. Consult agronomists for real farming decisions.
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 12: SCRIPT EXECUTION GUARD
# ============================================================
# This ensures the main() function only runs when the script is
# executed directly (not when imported as a module).
if __name__ == "__main__":
    main()
