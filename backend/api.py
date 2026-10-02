"""FastAPI entry point for serving the existing saved churn pipelines."""

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from mysql.connector import Error as MySQLError, IntegrityError

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from backend.predictions import ChurnPredictor
from backend.schemas import PredictionRequest, PredictionResponse
from backend.database import DatabaseManager
from backend.schemas import AuthResponse, LoginRequest, SignupRequest, UserResponse
from backend.auth import (
    create_session_token,
    hash_password,
    hash_session_token,
    public_user,
    session_expiry,
    verify_password,
)
from backend.config import (
    AUTH_COOKIE_NAME,
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_SECURE,
    AUTH_SESSION_HOURS,
    DATA_DIR,
    TABLEAU_CONFIG,
)


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

configured_origins = [origin.strip() for origin in os.getenv(
    "FRONTEND_ORIGIN", ""
).split(",") if origin.strip()]
allowed_origins = list(dict.fromkeys([
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    *configured_origins,
]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are loaded once as the API process starts, not once per request.
predictor = ChurnPredictor()
database = DatabaseManager()


def get_current_user(request: Request) -> Dict[str, Any]:
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    try:
        user = database.get_user_by_session(hash_session_token(token))
    except MySQLError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service unavailable") from error
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=AUTH_SESSION_HOURS * 60 * 60,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        path="/",
    )


def read_csv(filename: str) -> pd.DataFrame:
    """Read an existing application CSV or return a useful API error."""
    path = DATA_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Data file is unavailable: {filename}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Data file could not be read: {filename}") from error


def records(frame: pd.DataFrame):
    """Convert pandas/numpy values to JSON-compatible row objects."""
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def model_comparison_data():
    return records(read_csv("model_comparison.csv"))


def feature_importance_data():
    return records(read_csv("feature_importance.csv"))


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


@app.post("/api/v1/auth/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(request: SignupRequest, response: Response):
    email = str(request.email).strip().lower()
    try:
        if database.get_user_by_email(email):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")
        user = database.create_user(request.name, email, hash_password(request.password))
        token = create_session_token()
        database.create_session(user["id"], hash_session_token(token), session_expiry())
    except HTTPException:
        raise
    except IntegrityError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists") from error
    except MySQLError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service unavailable") from error
    set_session_cookie(response, token)
    return {"message": "Account created successfully", "user": public_user(user)}


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login(request: LoginRequest, response: Response):
    email = str(request.email).strip().lower()
    try:
        user = database.get_user_by_email(email)
        if user is None or not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
        token = create_session_token()
        database.create_session(user["id"], hash_session_token(token), session_expiry())
    except HTTPException:
        raise
    except MySQLError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service unavailable") from error
    set_session_cookie(response, token)
    return {"message": "Login successful", "user": public_user(user)}


@app.get("/api/v1/auth/me", response_model=UserResponse)
def current_user(user: Dict[str, Any] = Depends(get_current_user)):
    return public_user(user)


@app.post("/api/v1/auth/logout")
def logout(request: Request, response: Response, user: Dict[str, Any] = Depends(get_current_user)):
    token = request.cookies.get(AUTH_COOKIE_NAME)
    try:
        database.delete_session(hash_session_token(token))
    except MySQLError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service unavailable") from error
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return {"message": "Logout successful"}


@app.get("/api/v1/predictions/history")
def prediction_history(page: int = 1, page_size: int = 20, user: Dict[str, Any] = Depends(get_current_user)):
    """Expose the existing paginated MySQL prediction history."""
    try:
        history, total = database.get_prediction_history(page=page, page_size=page_size)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Prediction history is unavailable") from error
    safe_page = max(1, page)
    safe_size = min(max(1, page_size), 100)
    return {
        "items": history,
        "pagination": {
            "page": safe_page,
            "page_size": safe_size,
            "total": total,
            "total_pages": (total + safe_size - 1) // safe_size,
        },
    }


@app.get("/api/v1/predictions/today")
def predictions_today(user: Dict[str, Any] = Depends(get_current_user)):
    """Expose the existing database helper's DB-server-day count."""
    try:
        return {"count": database.count_predictions_today()}
    except Exception as error:
        raise HTTPException(status_code=503, detail="Today's prediction count is unavailable") from error


@app.get("/api/v1/dashboard")
def dashboard_summary(user: Dict[str, Any] = Depends(get_current_user)):
    dataset = read_csv("churn_data.csv")
    comparison = read_csv("model_comparison.csv")
    best_model = None
    if {"Model", "Accuracy"}.issubset(comparison.columns):
        valid = comparison.copy()
        valid["Accuracy"] = pd.to_numeric(valid["Accuracy"], errors="coerce")
        valid = valid.dropna(subset=["Accuracy"])
        if not valid.empty:
            best_model = records(valid.loc[[valid["Accuracy"].idxmax()]])[0]
    return {
        "total_customers": len(dataset),
        "churn_rate": float((dataset["Churn"] == "Yes").mean() * 100) if "Churn" in dataset else 0.0,
        "best_model": best_model,
        "data_file_count": len(list(Path(DATA_DIR).glob("*.csv"))),
        "tableau_url": TABLEAU_CONFIG["url"],
    }


@app.get("/api/v1/dashboard/dataset-preview")
def dataset_preview(limit: int = 10, user: Dict[str, Any] = Depends(get_current_user)):
    dataset = read_csv("churn_data.csv")
    limit = min(max(1, limit), 100)
    return {"columns": list(dataset.columns), "total_rows": len(dataset), "rows": records(dataset.head(limit))}


@app.get("/api/v1/dashboard/model-performance")
def dashboard_model_performance(user: Dict[str, Any] = Depends(get_current_user)):
    return {"models": model_comparison_data()}


@app.get("/api/v1/dashboard/confusion-matrices")
def confusion_matrices(user: Dict[str, Any] = Depends(get_current_user)):
    return {
        "logistic": records(read_csv("cm_logistic_tableau.csv")),
        "random_forest": records(read_csv("cm_rf_tableau.csv")),
        "deep_learning": records(read_csv("cm_deep_learning_tableau.csv")),
    }


@app.get("/api/v1/dashboard/feature-importance")
def dashboard_feature_importance(user: Dict[str, Any] = Depends(get_current_user)):
    return {"features": feature_importance_data()}


@app.get("/api/v1/analytics")
def analytics_summary(user: Dict[str, Any] = Depends(get_current_user)):
    predictions = read_csv("final_predictions.csv")
    return {
        "customers_analyzed": len(predictions),
        "high_risk_customers": int((predictions["Risk_Level"] == "High Risk").sum()) if "Risk_Level" in predictions else 0,
        "average_churn_probability": float(predictions["Churn_Probability"].mean() * 100) if "Churn_Probability" in predictions else 0.0,
        "risk_label_system": "Training analytics: <0.30 Low Risk, <0.60 Medium Risk, otherwise High Risk.",
    }


@app.get("/api/v1/analytics/risk-distribution")
def analytics_risk_distribution(user: Dict[str, Any] = Depends(get_current_user)):
    predictions = read_csv("final_predictions.csv")
    if "Risk_Level" not in predictions:
        return {"risk_label_system": "Training analytics", "items": []}
    counts = predictions["Risk_Level"].value_counts().rename_axis("risk_level").reset_index(name="count")
    return {"risk_label_system": "Training analytics", "items": records(counts)}


@app.get("/api/v1/analytics/model-comparison")
def analytics_model_comparison(user: Dict[str, Any] = Depends(get_current_user)):
    return {"models": model_comparison_data()}


@app.get("/api/v1/analytics/feature-importance")
def analytics_feature_importance(user: Dict[str, Any] = Depends(get_current_user)):
    return {"features": feature_importance_data()}


@app.post("/api/v1/predictions/{model_type}", response_model=PredictionResponse)
def create_prediction(
    model_type: str,
    request: PredictionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
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
