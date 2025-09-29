from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("model.joblib")
app = FastAPI()


class Input(BaseModel):
    features: list


@app.post("/predict")
def predict(data: Input):
    X = np.array(data.features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    return {
        "heart_disease_probability": float(proba),
        "prediction": int(proba > 0.5),
    }
