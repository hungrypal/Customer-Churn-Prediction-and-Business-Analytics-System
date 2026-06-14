import pandas as pd
import mysql.connector
from pathlib import Path
from backend.config import DB_CONFIG

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv(Path(__file__).resolve().parent / "data" / "churn_data.csv")

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
    host=DB_CONFIG["host"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    database=DB_CONFIG["database"]
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS churn_raw_data (
    customerID VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(20),
    SeniorCitizen INT,
    Partner VARCHAR(10),
    Dependents VARCHAR(10),
    tenure INT,
    PhoneService VARCHAR(10),
    MultipleLines VARCHAR(30),
    InternetService VARCHAR(30),
    OnlineSecurity VARCHAR(30),
    OnlineBackup VARCHAR(30),
    DeviceProtection VARCHAR(30),
    TechSupport VARCHAR(30),
    StreamingTV VARCHAR(30),
    StreamingMovies VARCHAR(30),
    Contract VARCHAR(30),
    PaperlessBilling VARCHAR(10),
    PaymentMethod VARCHAR(50),
    MonthlyCharges DECIMAL(10,2),
    TotalCharges DECIMAL(10,2),
    Churn VARCHAR(10)
)
""")

# =========================
# 4. INSERT DATA
# =========================

for _, row in df.iterrows():

    cursor.execute("""
    INSERT INTO churn_raw_data (
        customerID, gender, SeniorCitizen, Partner, Dependents,
        tenure, PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges, Churn
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        gender = VALUES(gender),
        SeniorCitizen = VALUES(SeniorCitizen),
        Partner = VALUES(Partner),
        Dependents = VALUES(Dependents),
        tenure = VALUES(tenure),
        PhoneService = VALUES(PhoneService),
        MultipleLines = VALUES(MultipleLines),
        InternetService = VALUES(InternetService),
        OnlineSecurity = VALUES(OnlineSecurity),
        OnlineBackup = VALUES(OnlineBackup),
        DeviceProtection = VALUES(DeviceProtection),
        TechSupport = VALUES(TechSupport),
        StreamingTV = VALUES(StreamingTV),
        StreamingMovies = VALUES(StreamingMovies),
        Contract = VALUES(Contract),
        PaperlessBilling = VALUES(PaperlessBilling),
        PaymentMethod = VALUES(PaymentMethod),
        MonthlyCharges = VALUES(MonthlyCharges),
        TotalCharges = VALUES(TotalCharges),
        Churn = VALUES(Churn)
    """, tuple(row))

conn.commit()

print("Data imported successfully!")

cursor.close()
conn.close()