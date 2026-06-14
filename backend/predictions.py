import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List

from backend.config import (
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH
)


class ChurnPredictor:
    """
    Handle ML predictions using saved sklearn pipelines
    """

    def __init__(self):
        self.logistic_model = None
        self.rf_model = None
        self.load_models()

    def load_models(self):

        try:

            if Path(LOGISTIC_MODEL_PATH).exists():
                self.logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
                print("Logistic Pipeline Loaded")

            else:
                print("Logistic Pipeline Not Found")

            if Path(RANDOM_FOREST_MODEL_PATH).exists():
                self.rf_model = joblib.load(RANDOM_FOREST_MODEL_PATH)
                print("Random Forest Pipeline Loaded")

            else:
                print("Random Forest Pipeline Not Found")

        except Exception as e:
            print("Error Loading Models:", e)

    def prepare_features(self, features: Dict):

        df = pd.DataFrame([{
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": features["tenure"],
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": features["internet_service"],
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": features["contract_type"],
            "PaperlessBilling": "Yes",
            "PaymentMethod": features["payment_method"],
            "MonthlyCharges": features["monthly_charges"],
            "TotalCharges": features["total_charges"]
        }])

        return df

    def predict(self,
                features: Dict,
                model_type: str = "random_forest"):

        try:

            df = self.prepare_features(features)

            if model_type == "logistic":

                if self.logistic_model is None:
                    return {"error": "Logistic model not loaded"}

                model = self.logistic_model

            else:

                if self.rf_model is None:
                    return {"error": "Random Forest model not loaded"}

                model = self.rf_model

            prediction = model.predict(df)[0]

            probability = model.predict_proba(df)[0][1]

            return {

                "prediction": int(prediction),

                "probability": float(probability),

                "churn_risk":
                    "High" if probability > 0.70
                    else "Medium" if probability > 0.40
                    else "Low",

                "model_used": model_type

            }

        except Exception as e:

            return {
                "error": str(e),
                "prediction": None,
                "probability": None
            }

    def predict_batch(self,
                      features_list: List[Dict],
                      model_type: str = "random_forest"):

        results = []

        for features in features_list:

            results.append(
                self.predict(features, model_type)
            )

        return results