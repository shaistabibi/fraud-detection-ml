from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI App
app = FastAPI(
    title="Fraud Detection API",
    description="Machine Learning API for detecting fraudulent credit card transactions",
    version="1.0"
)

# Load Model
MODEL_PATH = "models/fraud_model.pkl"
model = joblib.load(MODEL_PATH)

# Input Data Schema
class Transaction(BaseModel):

    id: int

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float

# Root Endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
    }

# Prediction Endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    result = {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }

    if prediction == 1:
        result["message"] = "⚠️ Fraudulent Transaction Detected"
    else:
        result["message"] = "✅ Legitimate Transaction"

    return result