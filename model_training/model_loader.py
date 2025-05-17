import pickle
import os

MODEL_PATH = os.path.join("model_training", "best_model.pkl")
SCALER_PATH = os.path.join("model_training", "scaler.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler
 