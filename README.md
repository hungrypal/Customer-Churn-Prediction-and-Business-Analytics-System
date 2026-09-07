Customer Churn Prediction & Analytics System

Project Overview

This project focuses on predicting customer churn using machine learning and transforming predictions into actionable business insights through a dashboard.

## Prediction history database

Prediction history is stored in MySQL. Copy `.env.example` to `.env` and set
`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` in your deployment
environment. The `predictions` table is created automatically when the app can
connect. To migrate an existing `data/user_predictions.csv` once, run:

```bash
python scripts/migrate_prediction_history_csv.py
```

The goal is to help companies identify high-risk customers and reduce revenue loss.


Problem Statement

Customer churn significantly impacts business revenue.
This project builds a predictive model to identify customers likely to leave and supports decision-making through analytics.

 Tech Stack
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	Logistic Regression, Random Forest
	•	Tableau (Dashboard)
	•	VS Code


 Features
	•	Data Cleaning & Preprocessing
	•	Feature Engineering
	•	Model Training & Evaluation
	•	Churn Probability Prediction
	•	Business Dashboard (Customer Segmentation & Risk Analysis)


Model Performance
	•	Logistic Regression & Random Forest comparison
	•	Evaluation using Accuracy, Precision, Recall, F1-score
	•	Focus on Recall for churn detection

## Deployment configuration

The production API runs with:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port $PORT
```

Set `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, and
`FRONTEND_ORIGIN` in the deployment environment. The React static site must set
`VITE_API_URL` to the public backend URL at build time. `render.yaml` defines a
separate Render Web Service for FastAPI and Static Site for React.

The three saved `ml_model/*_pipeline.pkl` artifacts are required by the API and
are intentionally tracked so a clean Render build can load them.

## Prediction-threshold note

The saved models and runtime prediction behavior are unchanged. Training reports
binary evaluation results using a probability threshold above `0.40`, while the
runtime service uses each sklearn classifier's native `predict()` output and
labels risk using its existing `0.40` and `0.70` probability bands. These
thresholds should be aligned only through an explicitly approved model-policy
change, not as part of deployment configuration.
