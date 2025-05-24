import numpy as np
from model_training.model_loader import load_model, load_scaler 

model = load_model()
scaler = load_scaler()

def make_prediction(input_data: dict) -> dict:
    # Convert dict to array and reshape
    features = np.array([list(input_data.values())]).reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability_of_default": round(proba, 4)
    }

