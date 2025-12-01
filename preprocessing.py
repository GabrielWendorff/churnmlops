import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path="data/churn.csv"):
    print("📌 Carregando dados...")
    return pd.read_csv(path)

def preprocess_data(df):
    print("📌 Pré-processando dados...")

    # Renomear para garantir compatibilidade com sua API
    df.rename(columns={
        "CreditScore": "credit_score",
        "Geography": "country",
        "Gender": "gender",
        "Age": "age",
        "Tenure": "tenure",
        "Balance": "balance",
        "NumOfProducts": "products_number",
        "HasCrCard": "credit_card",
        "IsActiveMember": "active_member",
        "EstimatedSalary": "estimated_salary",
        "Exited": "churn"
    }, inplace=True)

    # FEATURES FIXAS — MESMAS DA API
    feature_cols = [
        "credit_score",
        "country",
        "gender",
        "age",
        "tenure",
        "balance",
        "products_number",
        "credit_card",
        "active_member",
        "estimated_salary"
    ]

    X = df[feature_cols].copy()
    y = df["churn"]

    # Label encode de texto
    le_country = LabelEncoder()
    le_gender = LabelEncoder()

    X["country"] = le_country.fit_transform(X["country"])
    X["gender"] = le_gender.fit_transform(X["gender"])

    # Escalar tudo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_country, le_gender
