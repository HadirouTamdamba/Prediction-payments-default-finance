# Purpose: Unit test for the FastAPI prediction endpoint using pytest

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_default():
    """
    Unit test to validate the /predict endpoint with a valid sample payload.
    Ensures the endpoint returns 200 OK and the expected keys.
    """
    payload = {
        "LIMIT_BAL": 20000.0,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "BILL_AMT1": 3913.0,
        "PAY_AMT1": 0.0,
        "PAY_AMT2": 689.0
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability_of_default" in response.json()

### Run this test : pytest test_api.py
