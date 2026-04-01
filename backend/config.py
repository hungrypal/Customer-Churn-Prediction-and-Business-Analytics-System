import os
from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_model"

# -----------------------------
# Database Configuration
# -----------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "015Rohit@"),
    "database": os.getenv("DB_NAME", "churn_project")
}

# -----------------------------
# Tableau Dashboard Link
# -----------------------------
TABLEAU_CONFIG = {
    "url": "https://public.tableau.com/views/ProjectStep1_17717871184790/Dashboard1?:embed=y"
}

# -----------------------------
# Model Paths
# -----------------------------
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"