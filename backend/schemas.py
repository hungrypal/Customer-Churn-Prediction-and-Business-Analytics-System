"""Validated API contracts for the existing churn model pipelines."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """The 19 raw features used to train every saved churn pipeline."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=18.25, le=118.75)
    TotalCharges: float = Field(ge=18.8, le=8684.8)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    churn_risk: Literal["Low", "Medium", "High"]
    model_used: Literal["logistic", "random_forest", "deep_learning"]
