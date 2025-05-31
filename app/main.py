from fastapi import FastAPI
from app.schemas import CreditInput
from app.predict import make_prediction

app = FastAPI(
    title="Credit Card Default Prediction API",
    description="API for predicting credit card default using a trained ML model.",
    version="1.0.0"
)

@app.get("/")
def welcome():
    return {"message": "Welcome to the Credit Default Prediction API!"}

@app.post("/predict")
def predict_default(data: CreditInput):
    prediction = make_prediction(data.dict())
    return prediction


#Application test : python -m uvicorn app.main:app --reload
