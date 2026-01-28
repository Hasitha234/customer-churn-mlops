import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from api import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Churn Prediction API" in data["message"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_predict_endpoint_valid_input():
    test_data = {
        "tenure": -0.101157,
        "MonthlyCharges": -1.502549,
        "TotalCharges": -0.722456,
        "gender_Male": 1,
        "Partner_Yes": 0,
        "Dependents_Yes": 1,
        "PhoneService_Yes": 0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 1,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 1,
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 1,
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 1,
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 1,
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 1,
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 1,
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 1,
        "Contract_One_year": 0,
        "Contract_Two_year": 1,
        "PaperlessBilling_Yes": 0,
        "PaymentMethod_Credit_card": 1,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 1,
        "SeniorCitizen_1": 0
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert data["prediction"] in ["WILL CHURN", "WILL STAY"]
    assert 0 <= data["churn_probability"] <= 1

def test_predict_endpoint_invalid_input():
    test_data = {
        "tenure": "invalid",
        "MonthlyCharges": -1.502549,
        "TotalCharges": -0.722456,
        "gender_Male": 1,
        "Partner_Yes": 0,
        "Dependents_Yes": 1,
        "PhoneService_Yes": 0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 1,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 1,
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 1,
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 1,
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 1,
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 1,
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 1,
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 1,
        "Contract_One_year": 0,
        "Contract_Two_year": 1,
        "PaperlessBilling_Yes": 0,
        "PaymentMethod_Credit_card": 1,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 1,
        "SeniorCitizen_1": 0
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 422

def test_prediction_batch_endpoint():
    test_data ={
        "customers": [
            {
                "tenure": -0.101157,
                "MonthlyCharges": -1.502549,
                "TotalCharges": -0.722456,
                "gender_Male": 1,
                "Partner_Yes": 0,
                "Dependents_Yes": 1,
                "PhoneService_Yes": 0,
                "MultipleLines_No_phone_service": 0,
                "MultipleLines_Yes": 1,
                "InternetService_Fiber_optic": 0,
                "InternetService_No": 1,
                "OnlineSecurity_No_internet_service": 0,
                "OnlineSecurity_Yes": 1,
                "OnlineBackup_No_internet_service": 0,
                "OnlineBackup_Yes": 1,
                "DeviceProtection_No_internet_service": 0,
                "DeviceProtection_Yes": 1,
                "TechSupport_No_internet_service": 0,
                "TechSupport_Yes": 1,
                "StreamingTV_No_internet_service": 0,
                "StreamingTV_Yes": 1,
                "StreamingMovies_No_internet_service": 0,
                "StreamingMovies_Yes": 1,
                "Contract_One_year": 0,
                "Contract_Two_year": 1,
                "PaperlessBilling_Yes": 0,
                "PaymentMethod_Credit_card": 1,
                "PaymentMethod_Electronic_check": 0,
                "PaymentMethod_Mailed_check": 1,
                "SeniorCitizen_1": 0
            }
        ]
    }

    response = client.post("/predict/batch", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1