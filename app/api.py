from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

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
    # Convertir a DataFrame con columnas esperadas
    input_dict = {
        "Age": [data.Age],
        "Sex": [data.Sex],
        "ChestPainType": [data.ChestPainType],
        "RestingBP": [data.RestingBP],
        "Cholesterol": [data.Cholesterol],
        "FastingBS": [data.FastingBS],
        "RestingECG": [data.RestingECG],
        "MaxHR": [data.MaxHR],
        "ExerciseAngina": [data.ExerciseAngina],
        "Oldpeak": [data.Oldpeak],
        "ST_Slope": [data.ST_Slope],
    }

    X = pd.DataFrame(input_dict)

    proba = model.predict_proba(X)[0][1]
    prediction = int(proba > 0.5)

    return {
        "heart_disease_probability": float(proba),
        "prediction": prediction
    }
