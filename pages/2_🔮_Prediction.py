import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.predictions import ChurnPredictor
from backend.database import DatabaseManager
from mysql.connector import Error as MySQLError

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
        customer_id = st.text_input("Customer ID *", placeholder="Enter customer ID")
        customer_left, customer_right = st.columns(2)
        with customer_left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
        with customer_right:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.markdown("### Services")
        service_left, service_right = st.columns(2)
        with service_left:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=1)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        with service_right:
            online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])

        st.markdown("### Contract & Billing")
        billing_left, billing_right = st.columns(2)
        with billing_left:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with billing_right:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.25, max_value=118.75, value=79.50, step=0.01)
            total_charges = st.number_input("Total Charges ($)", min_value=18.80, max_value=8684.80, value=1000.00, step=0.01)
        
        model_type = st.radio(
            "Select Model",
            ["random_forest", "logistic", "deep_learning"],
            horizontal=True
        )
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            if not customer_id:
                st.error("Please enter Customer ID")
            else:
                # These are exactly the 19 raw columns expected by every
                # persisted sklearn pipeline. No model feature is defaulted.
                features = {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
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


                    try:
                        st.session_state.db.save_prediction(
                            customer_id=customer_id,
                            model_used=model_type,
                            prediction=result["prediction"],
                            probability=result["probability"],
                            risk_level=result["churn_risk"],
                            features=features,
                        )
                    except MySQLError:
                        st.error("Prediction could not be recorded. Please try again later.")

                
            
            

                

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

st.markdown("---")
st.markdown("### 📋 Recent Predictions")

try:
    page_size = 20
    _, total_predictions = st.session_state.db.get_prediction_history(page_size=1)
    total_pages = max(1, (total_predictions + page_size - 1) // page_size)
    history_page = st.number_input(
        "History page", min_value=1, max_value=total_pages, value=1, step=1
    )
    history, _ = st.session_state.db.get_prediction_history(
        page=int(history_page), page_size=page_size
    )
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("No predictions made yet.")
except MySQLError:
    st.warning("Prediction history is temporarily unavailable.")
