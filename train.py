import pickle
import pandas as pd
from preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model():
    df = load_data()

    X_scaled, y, scaler, le_country, le_gender = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # salvar tudo
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "encoder_country": le_country,
            "encoder_gender": le_gender
        }, f)

    print("✅ Modelo treinado e salvo!")

if __name__ == "__main__":
    train_model()
