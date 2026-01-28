from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import traceback
from typing import List

app = FastAPI(title="Churn Prediction API",
              description="Predict customer churn using ML",
              version="1.0.0")

print("Loading ML model...")
mlflow.set_tracking_uri("./mlruns")
model_path = "./mlruns/2/models/m-474f8138c04c409dad89b99c0e57c765/artifacts/model.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Churn Prediction API",
            "model": type(model).__name__,
            "version": "2.0.0",
            "endpoints": {
                "predict": "/predict",
                "health": "/health",
                "docs": "/docs"
                },
                "ci_cd": "Automated with GitHub Actions"
    }

@app.get("/health")
def health():
    return {"status": "healthy",
            "model_loaded": model is not None,
            "model_type": type(model).__name__
            }

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int
    Contract_One_year: int
    Contract_Two_year: int
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
    SeniorCitizen_1: int

    @field_validator('gender_Male', 'Partner_Yes', 'Dependents_Yes', 
               'PhoneService_Yes', 'MultipleLines_No_phone_service',
               'MultipleLines_Yes', 'InternetService_Fiber_optic',
               'InternetService_No', 'OnlineSecurity_No_internet_service',
               'OnlineSecurity_Yes', 'OnlineBackup_No_internet_service',
               'OnlineBackup_Yes', 'DeviceProtection_No_internet_service',
               'DeviceProtection_Yes', 'TechSupport_No_internet_service',
               'TechSupport_Yes', 'StreamingTV_No_internet_service',
               'StreamingTV_Yes', 'StreamingMovies_No_internet_service',
               'StreamingMovies_Yes', 'Contract_One_year',
               'Contract_Two_year', 'PaperlessBilling_Yes',
               'PaymentMethod_Credit_card', 'PaymentMethod_Electronic_check',
               'PaymentMethod_Mailed_check', 'SeniorCitizen_1',
               mode='before')
    @classmethod
    def validate_binary(cls, v, field):
        if v not in [0, 1]:
            raise ValueError(f'{field.name} must be 0 or 1, got {v}')
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
        }

@app.post("/predict")
def predict(data: CustomerData):
    try:
        feature_names = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'gender_Male', 'Partner_Yes', 'Dependents_Yes',
            'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No internet service', 'TechSupport_Yes',
            'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes',
            'Contract_One year', 'Contract_Two year',
            'PaperlessBilling_Yes',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check', 'SeniorCitizen_1'
        ]

        input_data = pd.DataFrame([[
            data.tenure, data.MonthlyCharges, data.TotalCharges,
            data.gender_Male, data.Partner_Yes, data.Dependents_Yes,
            data.PhoneService_Yes, data.MultipleLines_No_phone_service,
            data.MultipleLines_Yes, data.InternetService_Fiber_optic,
            data.InternetService_No, data.OnlineSecurity_No_internet_service,
            data.OnlineSecurity_Yes, data.OnlineBackup_No_internet_service,
            data.OnlineBackup_Yes, data.DeviceProtection_No_internet_service,
            data.DeviceProtection_Yes, data.TechSupport_No_internet_service,
            data.TechSupport_Yes, data.StreamingTV_No_internet_service,
            data.StreamingTV_Yes, data.StreamingMovies_No_internet_service,
            data.StreamingMovies_Yes, data.Contract_One_year,
            data.Contract_Two_year, data.PaperlessBilling_Yes,
            data.PaymentMethod_Credit_card, data.PaymentMethod_Electronic_check,
            data.PaymentMethod_Mailed_check, data.SeniorCitizen_1
        ]], columns=feature_names)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        churn_prob = float(probability[1])
        if churn_prob > 0.7:
            risk_level = "HIGH"
        elif churn_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "success": True,
            "prediction": "WILL CHURN" if prediction == 1 else "WILL STAY",
            "churn_probability": round(churn_prob, 4),
            "stay_probability": round(float(probability[0]), 4),
            "risk_level": risk_level,
            "confidence": "High" if max(probability) > 0.8 else "Medium" if max(probability) > 0.6 else "Low"
        }
    
    except Exception as e:
        print(f" Prediction error: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
class BatchCustomerData(BaseModel):
    customers: List[CustomerData]
    
    class Config:
        json_schema_extra = {
            "example": {
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
                    },
                    {
                        "tenure": 1.5,
                        "MonthlyCharges": 0.8,
                        "TotalCharges": 1.2,
                        "gender_Male": 0,
                        "Partner_Yes": 1,
                        "Dependents_Yes": 1,
                        "PhoneService_Yes": 1,
                        "MultipleLines_No_phone_service": 0,
                        "MultipleLines_Yes": 0,
                        "InternetService_Fiber_optic": 1,
                        "InternetService_No": 0,
                        "OnlineSecurity_No_internet_service": 0,
                        "OnlineSecurity_Yes": 0,
                        "OnlineBackup_No_internet_service": 0,
                        "OnlineBackup_Yes": 1,
                        "DeviceProtection_No_internet_service": 0,
                        "DeviceProtection_Yes": 0,
                        "TechSupport_No_internet_service": 0,
                        "TechSupport_Yes": 1,
                        "StreamingTV_No_internet_service": 0,
                        "StreamingTV_Yes": 0,
                        "StreamingMovies_No_internet_service": 0,
                        "StreamingMovies_Yes": 1,
                        "Contract_One_year": 1,
                        "Contract_Two_year": 0,
                        "PaperlessBilling_Yes": 1,
                        "PaymentMethod_Credit_card": 0,
                        "PaymentMethod_Electronic_check": 1,
                        "PaymentMethod_Mailed_check": 0,
                        "SeniorCitizen_1": 1
                    }
                ]
            }
        }

@app.post("/predict/batch")
def predict_batch(data: BatchCustomerData):
    try:
        results = []
        
        for idx, customer in enumerate(data.customers):
            feature_names = [
                'tenure', 'MonthlyCharges', 'TotalCharges',
                'gender_Male', 'Partner_Yes', 'Dependents_Yes',
                'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
                'InternetService_Fiber optic', 'InternetService_No',
                'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                'TechSupport_No internet service', 'TechSupport_Yes',
                'StreamingTV_No internet service', 'StreamingTV_Yes',
                'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                'Contract_One year', 'Contract_Two year',
                'PaperlessBilling_Yes',
                'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                'PaymentMethod_Mailed check', 'SeniorCitizen_1'
            ]
            
            input_data = pd.DataFrame([[
                customer.tenure, customer.MonthlyCharges, customer.TotalCharges,
                customer.gender_Male, customer.Partner_Yes, customer.Dependents_Yes,
                customer.PhoneService_Yes, customer.MultipleLines_No_phone_service,
                customer.MultipleLines_Yes, customer.InternetService_Fiber_optic,
                customer.InternetService_No, customer.OnlineSecurity_No_internet_service,
                customer.OnlineSecurity_Yes, customer.OnlineBackup_No_internet_service,
                customer.OnlineBackup_Yes, customer.DeviceProtection_No_internet_service,
                customer.DeviceProtection_Yes, customer.TechSupport_No_internet_service,
                customer.TechSupport_Yes, customer.StreamingTV_No_internet_service,
                customer.StreamingTV_Yes, customer.StreamingMovies_No_internet_service,
                customer.StreamingMovies_Yes, customer.Contract_One_year,
                customer.Contract_Two_year, customer.PaperlessBilling_Yes,
                customer.PaymentMethod_Credit_card, customer.PaymentMethod_Electronic_check,
                customer.PaymentMethod_Mailed_check, customer.SeniorCitizen_1
            ]], columns=feature_names)
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            churn_prob = float(probability[1])
            
            results.append({
                "customer_index": idx,
                "prediction": "WILL CHURN" if prediction == 1 else "WILL STAY",
                "churn_probability": round(churn_prob, 4),
                "stay_probability": round(float(probability[0]), 4),
                "risk_level": "HIGH" if churn_prob > 0.7 else "MEDIUM" if churn_prob > 0.4 else "LOW"
            })
        
        return {
            "success": True,
            "total_customers": len(data.customers),
            "predictions": results,
            "summary": {
                "will_churn": sum(1 for r in results if r["prediction"] == "WILL CHURN"),
                "will_stay": sum(1 for r in results if r["prediction"] == "WILL STAY"),
                "high_risk": sum(1 for r in results if r["risk_level"] == "HIGH")
            }
        }
        
    except Exception as e:
        print(f" Batch prediction error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )