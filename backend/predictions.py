import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from backend.config import LOGISTIC_MODEL_PATH, RANDOM_FOREST_MODEL_PATH
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logistic_model = joblib.load(os.path.join(BASE_DIR, "ml_model/logistic_model.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "ml_model/random_forest_model.pkl"))

class ChurnPredictor:
    """Handle ML model predictions"""
    
    def __init__(self):
        self.logistic_model = None
        self.rf_model = None
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            if Path(LOGISTIC_MODEL_PATH).exists():
                self.logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
                print("Logistic Regression model loaded")
            else:
                print("Logistic Regression model not found")
            
            if Path(RANDOM_FOREST_MODEL_PATH).exists():
                self.rf_model = joblib.load(RANDOM_FOREST_MODEL_PATH)
                print("Random Forest model loaded")
            else:
                print("Random Forest model not found")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict(self, features: Dict, model_type: str = 'random_forest') -> Dict:
        """
        Make prediction for a single customer
        
        Args:
            features: Dictionary with customer features
            model_type: 'logistic' or 'random_forest'
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Select model
            if model_type == 'logistic' and self.logistic_model:
                model = self.logistic_model
            elif self.rf_model:
                model = self.rf_model
            else:
                return {'error': 'No model available', 'prediction': None, 'probability': None}
            
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Make prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]  # Probability of churn
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'churn_risk': 'High' if probability > 0.5 else 'Low',
                'model_used': model_type
            }
        except Exception as e:
            return {'error': str(e), 'prediction': None, 'probability': None}
    
    def predict_batch(self, features_list: List[Dict], 
                     model_type: str = 'random_forest') -> List[Dict]:
        """Make predictions for multiple customers"""
        results = []
        for features in features_list:
            result = self.predict(features, model_type)
            results.append(result)
        return results