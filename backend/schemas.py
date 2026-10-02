"""Validated API contracts for the existing churn model pipelines."""

from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


class SignupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name is required")
        return value

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return str(value).strip().lower()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return str(value).strip().lower()


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr


class AuthResponse(BaseModel):
    message: str
    user: UserResponse


class PredictionRequest(BaseModel):
    """Customer identifier + the 19 raw features for the churn pipelines.

    ``customerID`` is an identifier used for history/tracking only.
    It is **never** passed into the sklearn pipeline.
    """

    # ---- identifier (not an ML feature) ----
    customerID: str = Field(min_length=1)

    # ---- ML features (19) ----
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
