from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_endpoint():
    example_input = {
        "Age": 55,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 115.0,
        "Cholesterol": None,
        "FastingBS": 1,
        "RestingECG": "Normal",
        "MaxHR": 155,
        "ExerciseAngina": "N",
        "Oldpeak": 0.1,
        "ST_Slope": "Flat"
    }

    response = client.post("/predict", json=example_input)
    assert response.status_code == 200
    result = response.json()
    assert "heart_disease_probability" in result
    assert "prediction" in result