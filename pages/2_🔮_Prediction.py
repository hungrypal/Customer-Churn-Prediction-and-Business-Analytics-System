import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.predictions import ChurnPredictor
from backend.database import DatabaseManager
from backend.config import DATA_DIR
from datetime import datetime

st.set_page_config(
    page_title="Prediction - Churn Prediction",
    page_icon="🔮",
    layout="wide"
)

st.markdown('<p class="main-header">🔮 Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter customer details to predict churn probability</p>', unsafe_allow_html=True)

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = ChurnPredictor()

if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()

# Prediction form
col1, col2 = st.columns([2, 1])

with col1:
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("### Customer Information")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            customer_id = st.text_input("Customer ID *", placeholder="Enter customer ID")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=79.50)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.00)
        
        with col_b:
            contract_type = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
        
        model_type = st.radio(
            "Select Model",
            ["random_forest", "logistic"],
            horizontal=True
        )
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            if not customer_id:
                st.error("Please enter Customer ID")
            else:
                # Prepare features
                features = {
                    'customer_id': customer_id,
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'total_charges': total_charges,
                    'contract_type': contract_type,
                    'payment_method': payment_method,
                    'internet_service': internet_service
                }
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    result = st.session_state.predictor.predict(features, model_type)
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display result
                    st.session_state.last_prediction = result
                    st.session_state.last_prediction_features = features


                    # Save user prediction
                    prediction_record = pd.DataFrame([{
                        "customer_id": customer_id,
                        "model_used": model_type,
                        "prediction": result["prediction"],
                        "probability": result["probability"],
                        "risk_level": result["churn_risk"],
                        "timestamp": datetime.now()
                    }])

                    prediction_file = DATA_DIR / "user_predictions.csv"

                    if prediction_file.exists():
                        prediction_record.to_csv(
                            prediction_file,
                            mode="a",
                            header=False,
                            index=False
                        )
                    else:
                        prediction_record.to_csv(
                            prediction_file,
                            index=False
                        )
                    
                    # Save to database (optional)
                    # st.session_state.db.log_prediction(
                    #     customer_id, model_type, 
                    #     result['prediction'], result['probability'],
                    #     features
                    # )

                
            
            

                

with col2:
    if 'last_prediction' in st.session_state:
        result = st.session_state.last_prediction
        
        st.markdown("### Prediction Result")
        
        # Risk badge
        if result['churn_risk'] == 'High':
            st.error(f"🚨 {result['churn_risk']} Risk")
        else:
            st.success(f"✅ {result['churn_risk']} Risk")
        
        st.metric("Churn Probability", f"{result['probability']:.2%}")
        st.metric("Prediction", "Will Churn" if result['prediction'] == 1 else "Will Not Churn")
        st.metric("Model Used", result['model_used'].replace('_', ' ').title())
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "#d1fae5"},
                    {'range': [50, 75], 'color': "#fef3c7"},
                    {'range': [75, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔄 New Prediction"):
            del st.session_state.last_prediction
            st.rerun()

# Recent predictions (if using database)
st.markdown("---")
st.markdown("### 📋 Recent Predictions")

prediction_file = DATA_DIR / "user_predictions.csv"

if prediction_file.exists():
    predictions_df = pd.read_csv(prediction_file)

    st.dataframe(
        predictions_df.tail(10),
        use_container_width=True
    )

# You can load from database or CSV
# if (DATA_DIR / "final_predictions.csv").exists():
#     predictions_df = pd.read_csv(DATA_DIR / "final_predictions.csv")
    # st.dataframe(predictions_df.head(10), use_container_width=True)

