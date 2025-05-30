# import os
# import pandas as pd
# import requests
# import re
# from urllib.parse import urlparse

# # Set dataset path
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, "phishing_data.csv")

# # PhishTank API URL (Public API)
# PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"

# def fetch_phishing_urls():
#     """Fetch latest phishing URLs from PhishTank"""
#     try:
#         response = requests.get(PHISHTANK_URL, timeout=10)
#         if response.status_code == 200:
#             df = pd.read_csv(response.content.decode("utf-8"))
#             return df["url"].tolist()
#         else:
#             print("❌ Failed to fetch data from PhishTank")
#             return []
#     except Exception as e:
#         print(f"❌ Error fetching phishing data: {e}")
#         return []

# def extract_features(url):
#     """Extract features from a given URL"""
#     parsed_url = urlparse(url)
    
#     features = {
#         "url": url,
#         "length": len(url),
#         "has_https": 1 if parsed_url.scheme == "https" else 0,
#         "num_dots": url.count("."),
#         "num_hyphens": url.count("-"),
#         "subdomain_count": len(parsed_url.netloc.split(".")) - 2 if len(parsed_url.netloc.split(".")) > 2 else 0,
#         "is_ip_address": 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0,
#         "query_length": len(parsed_url.query),
#         "contains_login_keyword": 1 if any(keyword in url.lower() for keyword in ["login", "signin", "bank"]) else 0,
#         "label": "Phishing"
#     }
#     return features

# def update_dataset():
#     """Fetch new phishing URLs and update dataset"""
#     new_urls = fetch_phishing_urls()
#     if not new_urls:
#         print("⚠️ No new phishing URLs found.")
#         return
    
#     # Load existing dataset


import os
import pandas as pd
import requests
import re
import pickle
import numpy as np
from urllib.parse import urlparse
from tensorflow.keras.models import load_model

# Set dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/phishing_data.csv")

# PhishTank API URL (Public API)
PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"

# Load Trained ML Models
ML_MODEL_PATH = os.path.join(BASE_DIR, "../models/trained_ml.pkl")
DL_MODEL_PATH = os.path.join(BASE_DIR, "../models/trained_dl.h5")

# Load ML Models
with open(ML_MODEL_PATH, "rb") as file:
    models = pickle.load(file)
    rf_model = models["random_forest"]
    svm_model = models["svm"]
    xgb_model = models["xgboost"]
    scaler = models["scaler"]

# Load Deep Learning Model (LSTM)
dl_model = load_model(DL_MODEL_PATH)

def fetch_phishing_urls():
    """Fetch latest phishing URLs from PhishTank"""
    try:
        response = requests.get(PHISHTANK_URL, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(response.content.decode("utf-8"))
            return df["url"].tolist()
        else:
            print("❌ Failed to fetch data from PhishTank")
            return []
    except Exception as e:
        print(f"❌ Error fetching phishing data: {e}")
        return []

def extract_features(url):
    """Extract features from a given URL"""
    parsed_url = urlparse(url)
    
    features = {
        "length": len(url),
        "has_https": 1 if parsed_url.scheme == "https" else 0,
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "subdomain_count": len(parsed_url.netloc.split(".")) - 2 if len(parsed_url.netloc.split(".")) > 2 else 0,
        "is_ip_address": 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0,
        "query_length": len(parsed_url.query),
        "contains_login_keyword": 1 if any(keyword in url.lower() for keyword in ["login", "signin", "bank"]) else 0,
    }
    return features

def predict_url(url):
    """Predict URL using both ML and DL models"""
    # Extract features
    features = extract_features(url)
    feature_values = np.array(list(features.values())).reshape(1, -1)
    
    # Scale features using pre-trained scaler
    feature_values = scaler.transform(feature_values)

    # ML Model Predictions
    rf_pred = rf_model.predict(feature_values)[0]
    svm_pred = svm_model.predict(feature_values)[0]
    xgb_pred = xgb_model.predict(feature_values)[0]

    # DL Model Prediction (LSTM)
    dl_pred = dl_model.predict(feature_values)
    dl_pred = np.argmax(dl_pred)  # Convert probability output to class

    # Majority Voting for Final Prediction
    votes = [rf_pred, svm_pred, xgb_pred, dl_pred]
    final_prediction = max(set(votes), key=votes.count)  # Most frequent class

    # Label Mapping
    label_map = {0: "Safe", 1: "Phishing", 2: "Suspicious"}
    return label_map[final_prediction]

def update_dataset():
    """Fetch new phishing URLs, predict them, and update dataset"""
    new_urls = fetch_phishing_urls()
    if not new_urls:
        print("⚠️ No new phishing URLs found.")
        return

    new_data = []
    for url in new_urls:
        features = extract_features(url)
        features["url"] = url  # Add original URL for reference
        features["label"] = predict_url(url)  # Classify the URL
        new_data.append(features)

    # Convert to DataFrame
    new_df = pd.DataFrame(new_data)

    # Append to existing dataset
    if os.path.exists(DATASET_PATH):
        existing_df = pd.read_csv(DATASET_PATH)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    # Save updated dataset
    updated_df.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset updated with {len(new_urls)} new phishing URLs.")

if __name__ == "__main__":
    update_dataset()
