from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("model.joblib")
app = FastAPI()


class Input(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: float
    Cholesterol: float | None
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


@app.post("/predict")
def predict(data: Input):
    X = np.array([[
        data.Age, data.Sex, data.ChestPainType, data.RestingBP,
        data.Cholesterol, data.FastingBS, data.RestingECG, data.MaxHR,
        data.ExerciseAngina, data.Oldpeak, data.ST_Slope
    ]], dtype=object)

    proba = model.predict_proba(X)[0][1]
    return {"heart_disease_probability": proba, "prediction": int(proba > 0.5)}
