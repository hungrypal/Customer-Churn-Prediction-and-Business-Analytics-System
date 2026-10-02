import os
from pathlib import Path

from dotenv import load_dotenv

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_model"

# Explicitly load .env from project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Database Configuration
# -----------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "churn_session")
AUTH_SESSION_HOURS = max(1, int(os.getenv("AUTH_SESSION_HOURS", "24")))
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
if AUTH_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    AUTH_COOKIE_SAMESITE = "lax"

# -----------------------------
# Tableau Dashboard Link
# -----------------------------
TABLEAU_CONFIG = {
    "url": "https://public.tableau.com/views/ProjectStep1_17717871184790/Dashboard1?:embed=y"
}

# -----------------------------
# Model Paths
# -----------------------------
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_pipeline.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.pkl"
DEEP_LEARNING_MODEL_PATH = MODELS_DIR / "deep_learning_pipeline.pkl"
