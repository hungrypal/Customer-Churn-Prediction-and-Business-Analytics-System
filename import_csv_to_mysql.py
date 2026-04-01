import pandas as pd
import mysql.connector

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/churn_data.csv")

# =========================
# 2. CLEAN DATA
# =========================

# convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# =========================
# 3. CONNECT TO MYSQL
# =========================

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="015Rohit@",
    database="churn_project"
)

cursor = conn.cursor()

# =========================
# 4. INSERT DATA
# =========================

for _, row in df.iterrows():

    cursor.execute("""
    INSERT INTO customers (
        customerID, gender, SeniorCitizen, Partner, Dependents,
        tenure, PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges, Churn
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, tuple(row))

conn.commit()

print("Data imported successfully!")

cursor.close()
conn.close()