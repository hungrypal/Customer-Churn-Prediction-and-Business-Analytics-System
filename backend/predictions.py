import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List

from backend.config import (
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    DEEP_LEARNING_MODEL_PATH
)


MODEL_FEATURES = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
)


class ChurnPredictor:
    """
    Handle ML predictions using saved sklearn pipelines
    """

    def __init__(self):
        self.logistic_model = None
        self.rf_model = None
        self.deep_learning_model = None
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

            if Path(DEEP_LEARNING_MODEL_PATH).exists():
                self.deep_learning_model = joblib.load(DEEP_LEARNING_MODEL_PATH)
                print("Deep Learning Pipeline Loaded")

            else:
                print("Deep Learning Pipeline Not Found")

        except Exception as e:
            print("Error Loading Models:", e)

    def prepare_features(self, features: Dict):
        """Return the exact raw columns expected by the trained pipelines.

        Required fields are intentionally not defaulted. The caller must provide
        all 19 features so that predictions represent the supplied customer.
        """
        missing = [feature for feature in MODEL_FEATURES if feature not in features]
        if missing:
            raise ValueError(f"Missing required model features: {', '.join(missing)}")

        return pd.DataFrame([{feature: features[feature] for feature in MODEL_FEATURES}])

    def predict(self,
                features: Dict,
                model_type: str = "random_forest"):

        try:

            df = self.prepare_features(features)

            if model_type == "logistic":

                if self.logistic_model is None:
                    return {"error": "Logistic model not loaded"}

                model = self.logistic_model

            elif model_type == "deep_learning":

                if self.deep_learning_model is None:
                    return {"error": "Deep learning model not loaded. Run train_model.py first."}

                model = self.deep_learning_model

            elif model_type == "random_forest":

                if self.rf_model is None:
                    return {"error": "Random Forest model not loaded"}

                model = self.rf_model

            else:
                return {"error": f"Unknown model type: {model_type}"}

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
