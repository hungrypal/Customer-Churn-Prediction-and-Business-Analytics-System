"""FastAPI entry point for serving the existing saved churn pipelines."""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.predictions import ChurnPredictor
from backend.schemas import PredictionRequest, PredictionResponse
from backend.database import DatabaseManager


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

allowed_origins = [origin.strip() for origin in os.getenv(
    "FRONTEND_ORIGIN", ""
).split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Models are loaded once as the API process starts, not once per request.
predictor = ChurnPredictor()
database = DatabaseManager()


@app.on_event("startup")
def initialize_database():
    """Attempt initialization without preventing a controlled API startup."""
    database.initialize()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "logistic": predictor.logistic_model is not None,
            "random_forest": predictor.rf_model is not None,
            "deep_learning": predictor.deep_learning_model is not None,
        },
    }


@app.get("/api/v1/models")
def available_models():
    return [
        {"id": "logistic", "name": "Logistic Regression"},
        {"id": "random_forest", "name": "Random Forest"},
        {"id": "deep_learning", "name": "Deep Learning"},
    ]


@app.post("/api/v1/predictions/{model_type}", response_model=PredictionResponse)
def create_prediction(
    model_type: str,
    request: PredictionRequest,
):
    if model_type not in {"logistic", "random_forest", "deep_learning"}:
        raise HTTPException(status_code=404, detail="Unknown model type")

    # Separate the identifier from the ML features.
    customer_id = request.customerID
    ml_features = {
        k: v for k, v in request.model_dump().items() if k != "customerID"
    }

    result = predictor.predict(ml_features, model_type)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    try:
        database.save_prediction(
            customer_id=customer_id,
            model_used=result["model_used"],
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["churn_risk"],
            features=ml_features,
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Prediction could not be recorded. Please try again later.",
        )
    return result
