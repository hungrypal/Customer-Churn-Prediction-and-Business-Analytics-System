import joblib
import os

def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    logistic = joblib.load(os.path.join(base_dir, "logistic_model.pkl"))
    rf = joblib.load(os.path.join(base_dir, "random_forest_model.pkl"))

    return logistic, rf