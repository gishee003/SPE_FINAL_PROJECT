import pytest
import requests

# URLs (Assuming Docker Compose service names or localhost for local testing)
INGEST_URL = "http://localhost:5000/ingest"
TRAIN_URL = "http://localhost:5001/train"
PREDICT_URL = "http://localhost:5002/predict"
DRIFT_URL = "http://localhost:5003/drift"

@pytest.fixture(scope="module", autouse=True)
def setup_initial_model():
    """Ensure a model exists before testing ingestion/prediction."""
    sample_data = [
        {
            "CustomerId": 1,
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Male",
            "Age": 30,
            "Tenure": 5,
            "Balance": 1000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 40000.0,
            "Exited": 0
        },
        {
            "CustomerId": 2,
            "CreditScore": 650,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 45,
            "Tenure": 1,
            "Balance": 5000.0,
            "NumOfProducts": 2,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 50000.0,
            "Exited": 1
        }
    ]
    requests.post(TRAIN_URL, json=sample_data)

def test_full_pipeline_flow():
    # 1. Test Ingestion (which triggers Predict and Drift)
    payload = [
        {
            "CustomerId": 101,
            "CreditScore": 700,
            "Geography": "Spain",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 3,
            "Balance": 2500.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 45000.0,
            "Exited": 0
        }
    ]
    response = requests.post(INGEST_URL, json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "serving_response" in data
    assert "drift_response" in data

def test_drift_triggers_retraining():
    # 2. Simulate Drift (Drastic change in Age/Balance)
    drift_payload = [
        {
            "CustomerId": 999,
            "CreditScore": 300,
            "Geography": "France",
            "Gender": "Male",
            "Age": 99,
            "Tenure": 50,
            "Balance": 999999.0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 100000.0,
            "Exited": 1
        }
    ]
    
    # Call Drift service directly
    response = requests.post(DRIFT_URL, json=drift_payload)
    assert response.status_code == 200
    
    # If drift is detected, it should return training_response
    if response.json().get("drift_detected"):
        assert "training_response" in response.json()
        assert response.json()["training_response"]["status"] in ["success", "retraining_started"]
