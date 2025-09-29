from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_endpoint():
    # Ejemplo con las 11 columnas originales
    example_input = {
        "features": [
            55,             # Age
            "M",            # Sex
            "ASY",          # ChestPainType
            115.0,          # RestingBP
            None,           # Cholesterol (puede venir NaN)
            1,              # FastingBS
            "Normal",       # RestingECG
            155,            # MaxHR
            "N",            # ExerciseAngina
            0.1,            # Oldpeak
            "Flat"          # ST_Slope
        ]
    }

    response = client.post("/predict", json=example_input)
    assert response.status_code == 200
    result = response.json()
    assert "heart_disease_probability" in result
    assert "prediction" in result