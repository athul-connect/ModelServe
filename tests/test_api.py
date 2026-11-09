from fastapi.testclient import TestClient
from app.api import app
from unittest.mock import patch

client = TestClient(app)

def test_invalid_tenture():
    response = client.post(
        "/predict",
        json = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": -10,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 425.0
        }
    )
    assert response.status_code == 422


def test_valid_customer_prediction():
    customer = {
         "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 425.0
    }
    response = client.post("/predict",json= customer)

    assert response.status_code == 200

    result = response.json()

    assert result["churn_prediction"] in ["yes","No"]
    assert 0 <= result ["churn_probability"] <= 1

def test_prediction_failure():
    with patch(
        "app.api.model.predict",
        side_effect = RuntimeError("simulates model failure")
    ):
        response = client.post(
            "/predict",
            json={
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 425.0
            }
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "prediction failed"