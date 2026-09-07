
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend.config import DATA_DIR, TABLEAU_CONFIG
from backend.predictions import ChurnPredictor
from backend.database import DatabaseManager

import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    logistic_model = joblib.load(os.path.join(BASE_DIR, "ml_model/logistic_model.pkl"))
    rf_model = joblib.load(os.path.join(BASE_DIR, "ml_model/random_forest_model.pkl"))
except:
    logistic_model = None
    rf_model = None

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = ChurnPredictor()

if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3153/3153420.png", width=100)
    st.title("ChurnPredict")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Dashboard", "🔮 Prediction", "📈 Analytics"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.info("✅ Logistic Regression\n✅ Random Forest\n✅ Deep Learning Neural Network")
    
    st.markdown("---")
    st.markdown("### Data Source")
    st.success(f"📁 {len(list(DATA_DIR.glob('*.csv')))} CSV files loaded")

# Home Page
if menu == "🏠 Home":
    st.markdown('<p class="main-header">Customer Churn Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-powered analytics to predict and prevent customer churn</p>', unsafe_allow_html=True)

    # Load dataset for metrics
    df = None
    best_model_name = None
    best_model_accuracy = None

    if (DATA_DIR / "churn_data.csv").exists():
        df = pd.read_csv(DATA_DIR / "churn_data.csv")

        total_customers = len(df)

        churn_rate = 0
        if "Churn" in df.columns:
            churn_rate = df["Churn"].value_counts(normalize=True).get(1, 0) * 100

    else:
        total_customers = 0
        churn_rate = 0

    model_comparison_file = DATA_DIR / "model_comparison.csv"
    if model_comparison_file.exists():
        try:
            model_comparison = pd.read_csv(model_comparison_file)
            if {"Model", "Accuracy"}.issubset(model_comparison.columns):
                model_comparison["Accuracy"] = pd.to_numeric(
                    model_comparison["Accuracy"], errors="coerce"
                )
                best_model = model_comparison.dropna(subset=["Accuracy"]).loc[
                    lambda frame: frame["Accuracy"].idxmax()
                ]
                best_model_name = best_model["Model"]
                best_model_accuracy = float(best_model["Accuracy"])
        except (OSError, ValueError, KeyError):
            pass
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", total_customers)

    with col2:
        st.metric("Churn Rate", f"{churn_rate:.2f}%")

    with col3:
        if best_model_accuracy is not None:
            st.metric("Best Model Accuracy", f"{best_model_accuracy:.1%}")
            st.caption(best_model_name)
        else:
            st.metric("Best Model Accuracy", "Not available")

    with col4:

        try:
            prediction_count = st.session_state.db.count_predictions_today()
        except Exception:
            prediction_count = 0

        st.metric(
            "Predictions Today",
            prediction_count
        )
    st.markdown("---")

    st.markdown("### 📋 Recent Predictions")
    try:
        recent_predictions, _ = st.session_state.db.get_prediction_history(page_size=10)
    except Exception:
        recent_predictions = []
        st.warning("Prediction history is temporarily unavailable.")
    if recent_predictions:
        st.dataframe(pd.DataFrame(recent_predictions), use_container_width=True)
    else:
        st.info("No predictions made yet.")

    
    # Features
    st.markdown("### 🔑 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("#### 🤖 Machine Learning Models")
            st.markdown("""
            - Logistic Regression
            - Random Forest
            - Deep Learning Neural Network
            - Real-time predictions
            """)
    
    with col2:
        with st.container():
            st.markdown("#### 📊 Visualizations")
            st.markdown("""
            - Interactive Tableau dashboards
            - Confusion matrices
            - Feature importance
            - Model comparison
            """)
    
    # Load sample data preview
    st.markdown("---")
    st.markdown("### 📂 Data Preview")
    
    if (DATA_DIR / "churn_data.csv").exists():
        df = pd.read_csv(DATA_DIR / "churn_data.csv", nrows=5)
        st.dataframe(df, use_container_width=True)
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.switch_page("pages/1_📊_Dashboard.py")
    
    with col2:
        if st.button("🔮 Make Prediction", use_container_width=True):
            st.switch_page("pages/2_🔮_Prediction.py")
    
    with col3:
        if st.button("📈 View Analytics", use_container_width=True):
            st.switch_page("pages/3_📈_Analytics.py")

# You can add more pages as needed
elif menu == "📊 Dashboard":
    st.switch_page("pages/1_📊_Dashboard.py")

elif menu == "🔮 Prediction":
    st.switch_page("pages/2_🔮_Prediction.py")

elif menu == "📈 Analytics":
    st.markdown("### Analytics Page - Coming Soon")
