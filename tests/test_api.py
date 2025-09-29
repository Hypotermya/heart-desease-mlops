from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_predict_endpoint():
    # Ejemplo de features (ajústalo al tamaño esperado por tu modelo)
    example_input = {
        "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    }

    response = client.post("/predict", json=example_input)
    
    # Verificar que el endpoint responde bien
    assert response.status_code == 200

    result = response.json()
    assert "heart_disease_probability" in result
    assert "prediction" in result
    assert 0.0 <= result["heart_disease_probability"] <= 1.0
    assert result["prediction"] in [0, 1]