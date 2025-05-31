import numpy as np
from model_training.model_loader import load_model, load_scaler 

model = load_model()
scaler = load_scaler()

# Order of features expected by the model (must match training phase)
expected_features = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"
]

def make_prediction(input_data: dict) -> dict:
    # Ensure correct feature order
    features = np.array([[input_data[feature] for feature in expected_features]])
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability_of_default": round(proba, 4)
    }
