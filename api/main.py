from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="Churn Prediction API",
    description="Modelo de Regressão Logística para Churn",
    version="1.0"
)

# ============================
# CORS
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ou ["http://127.0.0.1:5500"] se quiser restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_form():
    return FileResponse(os.path.join(STATIC_DIR, "form.html"))

# ============================
# Carregar modelo + scaler + encoders
# ============================
bundle = joblib.load("model/model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
encoder_country = bundle["encoder_country"]
encoder_gender = bundle["encoder_gender"]


# ============================
# Schema da Requisição
# ============================
class ChurnRequest(BaseModel):
    credit_score: float
    country: str
    gender: str
    age: float
    tenure: float
    balance: float
    products_number: float
    credit_card: int
    active_member: int
    estimated_salary: float


# ============================
# Rota de Previsão
# ============================
@app.post("/predict")
def predict_churn(data: ChurnRequest):

    # Aplicar label encoding da mesma forma que no treino
    try:
        country_encoded = encoder_country.transform([data.country])[0]
    except ValueError:
        return {"erro": f"País '{data.country}' não existe no treinamento"}

    try:
        gender_encoded = encoder_gender.transform([data.gender])[0]
    except ValueError:
        return {"erro": f"Gênero '{data.gender}' não existe no treinamento"}

    # Criar array na ordem correta
    input_data = np.array([[
        data.credit_score,
        country_encoded,
        gender_encoded,
        data.age,
        data.tenure,
        data.balance,
        data.products_number,
        data.credit_card,
        data.active_member,
        data.estimated_salary
    ]])

    # Escalar com o scaler treinado
    scaled = scaler.transform(input_data)

    # Predizer
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    return {
        "churn_predito": int(pred),
        "probabilidade_de_churn": round(float(prob), 4)
    }
