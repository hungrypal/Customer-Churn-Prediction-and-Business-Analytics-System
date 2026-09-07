from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_model"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ===============================
# 1. LOAD DATA
# ===============================

df = pd.read_csv(DATA_DIR / "churn_data.csv")

print("Columns:")
print(df.columns)

print("\nShape of Dataset:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# ===============================
# 2. DATA CLEANING
# ===============================

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values using median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Save customer IDs
customer_ids = df["customerID"]

# Convert churn to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# ===============================
# 3. FEATURE ENGINEERING
# ===============================

# Remove ID column
df_model = df.drop("customerID", axis=1)



# Separate features and labels
X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            # A dense output works with the neural network and is still small
            # enough for this tabular churn dataset.
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        )
    ]
)


# ===============================
# 4. TRAIN TEST SPLIT
# ===============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


# ===============================
# 5. FEATURE SCALING
# ===============================

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# ===============================
# 6. MODEL TRAINING
# ===============================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Logistic Regression
log_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        )
    )
])

log_pipeline.fit(X_train, y_train)

# Random Forest

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

rf_pipeline.fit(X_train, y_train)

# Deep Learning (multilayer neural network)
# Scaling numeric features and one-hot encoding categoricals happen in the
# pipeline before the network is trained.
deep_learning_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
    )
])

deep_learning_pipeline.fit(X_train, y_train)


# ===============================
# 7. MODEL PREDICTIONS
# ===============================

# Logistic predictions
log_prob = log_pipeline.predict_proba(X_test)[:, 1]
log_pred = (log_prob > 0.4).astype(int)

# Random forest predictions
rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob > 0.4).astype(int)

# Deep learning predictions
deep_prob = deep_learning_pipeline.predict_proba(X_test)[:, 1]
deep_pred = (deep_prob > 0.4).astype(int)


# ===============================
# 8. MODEL EVALUATION
# ===============================

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    roc_auc_score
)

print("\n===== Logistic Regression Results =====")

print("Accuracy:", accuracy_score(y_test, log_pred))
print("ROC AUC:", roc_auc_score(y_test, log_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))

print("\nClassification Report:")
print(classification_report(y_test, log_pred))


print("\n===== Random Forest Results =====")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\n===== Deep Learning Neural Network Results =====")

print("Accuracy:", accuracy_score(y_test, deep_pred))
print("ROC AUC:", roc_auc_score(y_test, deep_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, deep_pred))

print("\nClassification Report:")
print(classification_report(y_test, deep_pred))


# ===============================
# 9. FEATURE IMPORTANCE
# ===============================

# feature_importance = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": rf_model.feature_importances_
# })

# feature_importance = feature_importance.sort_values(
#     by="Importance",
#     ascending=False
# )

# print("\nTop Important Features:")
# print(feature_importance.head(10))

# feature_importance.to_csv(DATA_DIR / "feature_importance.csv", index=False)


# ===============================
# 10. MODEL COMPARISON
# ===============================

model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Deep Learning Neural Network"],
    "Accuracy": [
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, deep_pred)
    ],
    "Recall_Churn": [
        recall_score(y_test, log_pred),
        recall_score(y_test, rf_pred),
        recall_score(y_test, deep_pred)
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, log_prob),
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, deep_prob)
    ]
})

model_comparison.to_csv(DATA_DIR / "model_comparison.csv", index=False)


# ===============================
# 11. CONFUSION MATRICES (TABLEAU)
# ===============================

cm_log = confusion_matrix(y_test, log_pred)

cm_log_tableau = pd.DataFrame({
    "Actual": [0, 0, 1, 1],
    "Predicted": [0, 1, 0, 1],
    "Count": cm_log.flatten()
})

cm_log_tableau.to_csv(DATA_DIR / "cm_logistic_tableau.csv", index=False)


cm_rf = confusion_matrix(y_test, rf_pred)

cm_rf_tableau = pd.DataFrame({
    "Actual": [0, 0, 1, 1],
    "Predicted": [0, 1, 0, 1],
    "Count": cm_rf.flatten()
})

cm_rf_tableau.to_csv(DATA_DIR / "cm_rf_tableau.csv", index=False)


cm_deep = confusion_matrix(y_test, deep_pred)

cm_deep_tableau = pd.DataFrame({
    "Actual": [0, 0, 1, 1],
    "Predicted": [0, 1, 0, 1],
    "Count": cm_deep.flatten()
})

cm_deep_tableau.to_csv(DATA_DIR / "cm_deep_learning_tableau.csv", index=False)


# ===============================
# 12. RISK SEGMENTATION
# ===============================

def risk_segment(prob):

    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


# ===============================
# 13. SAVE FINAL PREDICTIONS
# ===============================

results = X_test.copy()

results["CustomerID"] = customer_ids.loc[X_test.index]
results["Actual_Churn"] = y_test
results["Predicted_Churn"] = log_pred
results["Churn_Probability"] = log_prob
results["Risk_Level"] = results["Churn_Probability"].apply(risk_segment)

results["RF_Predicted_Churn"] = rf_pred
results["RF_Churn_Probability"] = rf_prob

results["Deep_Learning_Predicted_Churn"] = deep_pred
results["Deep_Learning_Churn_Probability"] = deep_prob

results.to_csv(DATA_DIR / "final_predictions.csv", index=False)

print("\nFinal prediction file saved successfully.")


# ===============================
# 14. SAVE MODELS
# ===============================

joblib.dump(
    log_pipeline,
    MODELS_DIR / "logistic_pipeline.pkl"
)

joblib.dump(
    rf_pipeline,
    MODELS_DIR / "random_forest_pipeline.pkl"
)

joblib.dump(
    deep_learning_pipeline,
    MODELS_DIR / "deep_learning_pipeline.pkl"
)

print("Pipeline models saved successfully")

print("\nModels saved successfully.")
