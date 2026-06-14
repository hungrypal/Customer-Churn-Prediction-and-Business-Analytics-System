import streamlit as st
import pandas as pd
import plotly.express as px
from backend.config import DATA_DIR


st.set_page_config(
    page_title="Analytics - Customer Churn System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Business Analytics")
st.subheader("Churn risk, model performance, and action planning")


def load_csv(filename: str):
    file_path = DATA_DIR / filename
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


final_predictions = load_csv("final_predictions.csv")
model_comparison = load_csv("model_comparison.csv")
feature_importance = load_csv("feature_importance.csv")

col1, col2, col3 = st.columns(3)

with col1:
    total_customers = len(final_predictions) if final_predictions is not None else 0
    st.metric("Customers Analyzed", total_customers)

with col2:
    high_risk = 0
    if final_predictions is not None and "Risk_Level" in final_predictions.columns:
        high_risk = int((final_predictions["Risk_Level"] == "High Risk").sum())
    st.metric("High-Risk Customers", high_risk)

with col3:
    avg_probability = 0.0
    if final_predictions is not None and "Churn_Probability" in final_predictions.columns:
        avg_probability = float(final_predictions["Churn_Probability"].mean() * 100)
    st.metric("Average Churn Probability", f"{avg_probability:.1f}%")

st.markdown("---")

left, right = st.columns(2)

with left:
    st.markdown("### Risk Distribution")
    if final_predictions is not None and "Risk_Level" in final_predictions.columns:
        risk_counts = final_predictions["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `train_model.py` first to generate risk segmentation outputs.")

with right:
    st.markdown("### Model Comparison")
    if model_comparison is not None:
        metrics_cols = [col for col in ["Accuracy", "Recall_Churn", "ROC_AUC"] if col in model_comparison.columns]
        if metrics_cols:
            fig = px.bar(
                model_comparison,
                x="Model",
                y=metrics_cols,
                barmode="group",
                title="Model Performance Overview"
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(model_comparison, use_container_width=True)
    else:
        st.info("Run `train_model.py` to generate model comparison metrics.")

st.markdown("---")
st.markdown("### Feature Importance")
if feature_importance is not None:
    top_features = feature_importance.head(10)
    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Drivers of Churn"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_features, use_container_width=True)
else:
    st.info("Run `train_model.py` to generate feature importance data.")

st.markdown("---")
st.markdown("### Action Plan")
st.write(
    """
    - Prioritize outreach to high-risk customers before contract renewal.
    - Use tenure, monthly charges, and contract type to target retention campaigns.
    - Compare Logistic Regression vs Random Forest to choose the best production model.
    - Feed new predictions into the database to track churn trends over time.
    """
)