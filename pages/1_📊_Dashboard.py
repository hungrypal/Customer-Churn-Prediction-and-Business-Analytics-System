import streamlit as st
import pandas as pd
import plotly.express as px
from backend.config import DATA_DIR, TABLEAU_CONFIG

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Dashboard")
st.subheader("Machine Learning Model Performance & Business Insights")

# ----------------------------------
# Load Data
# ----------------------------------

confusion_matrix_logistic = None
confusion_matrix_rf = None
feature_importance = None

try:
    confusion_matrix_logistic = pd.read_csv(DATA_DIR / "cm_logistic_tableau.csv")
except:
    pass

try:
    confusion_matrix_rf = pd.read_csv(DATA_DIR / "cm_rf_tableau.csv")
except:
    pass

try:
    feature_importance = pd.read_csv(DATA_DIR / "feature_importance.csv")
except:
    pass

# ----------------------------------
# Tableau Dashboard Section
# ----------------------------------

st.markdown("---")
st.header("🎯 Tableau Business Dashboard")

st.write("Click below to open the interactive Tableau dashboard.")

st.link_button(
    "🔗 Open Tableau Dashboard",
    TABLEAU_CONFIG["url"]
)

# ----------------------------------
# Model Performance
# ----------------------------------

st.markdown("---")
st.header("📉 Model Performance")

col1, col2 = st.columns(2)

# Logistic Regression
with col1:
    st.subheader("Logistic Regression")

    if confusion_matrix_logistic is not None:

        cm_log = confusion_matrix_logistic.pivot(
            index="Actual",
            columns="Predicted",
            values="Count"
        )

        fig = px.imshow(
            cm_log,
            text_auto=True,
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Logistic confusion matrix file not found")


# Random Forest
with col2:
    st.subheader("Random Forest")

    if confusion_matrix_rf is not None:

        cm_rf = confusion_matrix_rf.pivot(
            index="Actual",
            columns="Predicted",
            values="Count"
        )

        fig = px.imshow(
            cm_rf,
            text_auto=True,
            color_continuous_scale="Greens"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Random Forest confusion matrix file not found")


# ----------------------------------
# Feature Importance
# ----------------------------------

st.markdown("---")
st.header("🎯 Feature Importance")

if feature_importance is not None:

    top_features = feature_importance.head(10)

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Feature importance file not found")


# ----------------------------------
# Dataset Preview
# ----------------------------------

st.markdown("---")
st.header("📄 Dataset Preview")

try:
    df = pd.read_csv(DATA_DIR / "churn_data.csv")
    st.dataframe(df.head(10), use_container_width=True)
except:
    st.info("Dataset preview not available")