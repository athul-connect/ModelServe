from fastapi import FastAPI , HTTPException
from pydantic import BaseModel,Field
import joblib
import pandas as pd 
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

app = FastAPI()

MODEL_PATH = (Path(__file__).resolve().parent.parent
              / "models"
              /"churn_pipeline.joblib"
              )
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found at: {MODEL_PATH}"
    )
model = joblib.load(MODEL_PATH)

@app.post("/Customer")
def receive_customer(customer:CustomerData):
    return customer.model_dump()

@app.post("/predict")
def predict_churn(customer:CustomerData):
    customer_data = customer.model_dump()
    customer_df = pd.DataFrame([customer_data])

    try:
        prediction = model.predict(customer_df)[0]
        probability = model.predict_proba(customer_df)[0][1]

    except Exception:
        logger.exception("prediction failed")
        raise HTTPException(
            status_code = 500,
            detail = "prediction failed"
        )

     
    return{
        "churn_prediction": "yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }