from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# -----------------------------
# Create FastAPI application
# -----------------------------

app = FastAPI(
    title="OpenHousing Price Prediction API",
    description="API for predicting median housing values",
    version="1.0.0"
)


# -----------------------------
# Load trained model
# -----------------------------

MODEL_PATH = "ML/model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------
# Input Schema
# -----------------------------

class HousingData(BaseModel):

    crim: float
    zn: float
    indus: float
    chas: float
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float


# -----------------------------
# Health Check Endpoint
# -----------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Random Forest",
        "version": "1.0"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict_price(data: HousingData):

    input_data = [[
        data.crim,
        data.zn,
        data.indus,
        data.chas,
        data.nox,
        data.rm,
        data.age,
        data.dis,
        data.rad,
        data.tax,
        data.ptratio,
        data.b,
        data.lstat
    ]]

    prediction = model.predict(input_data)

    return {
        "predicted_medv": round(
            float(prediction[0]),
            2
        ),
        "model_version": "1.0"
    }