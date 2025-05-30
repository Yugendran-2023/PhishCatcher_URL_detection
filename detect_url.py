import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from backend.feature_extraction import extract_features

# File paths
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
ML_MODEL_PATH = os.path.join(MODEL_DIR, "trained_ml.pkl")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "trained_dl.h5")

labels = ["Safe", "Phishing", "Suspicious"]

# Load ML models and scaler
with open(ML_MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

scaler = model_bundle["scaler"]
ml_model = model_bundle["RandomForest"]  # 👈 You can choose any: SVM, SGDClassifier, XGBoost

# Load DL model
dl_model = load_model(DL_MODEL_PATH)

def predict_url(url):
    try:
        features = extract_features(url)
        X = np.array([features])
        X_scaled = scaler.transform(X)

        # ML prediction
        ml_pred = ml_model.predict(X_scaled)[0]
        ml_proba = ml_model.predict_proba(X_scaled)[0]
        ml_label = labels[int(ml_pred)]

        # DL prediction
        dl_input = X_scaled.reshape((1, len(features), 1))
        dl_output = dl_model.predict(dl_input, verbose=0)[0]
        dl_label = labels[np.argmax(dl_output)]

        return {
            "url": url,
            "features": features.tolist(),
            "ml_model": {
                "prediction": ml_label,
                "confidence": {
                    labels[i]: round(ml_proba[i] * 100, 2) for i in range(len(labels))
                }
            },
            "dl_model": {
                "prediction": dl_label,
                "confidence": {
                    labels[i]: round(dl_output[i] * 100, 2) for i in range(len(labels))
                }
            }
        }
    except Exception as e:
        return {"error": str(e)}
