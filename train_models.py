import os
import pickle
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Bidirectional, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from feature_extraction import extract_features

LABEL_MAP = {"Safe": 0, "Phishing": 1, "Suspicious": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/phishing_data.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
os.makedirs(MODEL_DIR, exist_ok=True)

def get_safe_smote_k(y):
    """Determine a safe k_neighbors value for SMOTE based on smallest class size."""
    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    return max(1, min(min_class_size - 1, 5))  # Between 1 and 5


def train_ml():
    print("\n🔧 Training ML Models...")
    df = pd.read_csv(DATASET_PATH).dropna()
    df["label"] = df["label"].map(LABEL_MAP)

    if "url" in df.columns:
        X = df.drop(columns=["url", "label"])
    else:
        X = df.drop(columns=["label"])
    y = df["label"].values  # ✅ Now y is correctly initialized

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE
    print(f"📊 Class distribution before SMOTE: {Counter(y)}")
    smote = SMOTE(random_state=42, k_neighbors=get_safe_smote_k(y))
    X_res, y_res = smote.fit_resample(X_scaled, y)
    print(f"📊 Class distribution after SMOTE: {Counter(y_res)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": LinearSVC(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "SGDClassifier": SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    }

    for name, model in models.items():
        print(f"\n📚 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"✅ {name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=LABEL_MAP.keys()))

    with open(os.path.join(MODEL_DIR, "trained_ml.pkl"), "wb") as f:
        pickle.dump({**models, "scaler": scaler}, f)
    print("💾 ML models saved.")


def train_dl():
   
    print("\n🧠 Training DL Model...")
    df = pd.read_csv(DATASET_PATH).dropna()
    df["label"] = df["label"].map(LABEL_MAP)

    if "url" in df.columns:
        X = df.drop(columns=["url", "label"])
    else:
        X = df.drop(columns=["label"])
    y = df["label"]

    # Scale + SMOTE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42, k_neighbors=get_safe_smote_k(y))
    X_res, y_res = smote.fit_resample(X_scaled, y)
    y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y_res.to_numpy().reshape(-1, 1))
    X_reshaped = X_res.reshape(X_res.shape[0], X_res.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, stratify=np.argmax(y_encoded, axis=1), random_state=42)

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(128, 3, activation="relu"),
        Conv1D(128, 3, activation="relu"),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax")
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train, y_train, epochs=50, batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, factor=0.5)],
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "trained_dl.h5"))
    print("💾 DL model saved.")

def train_all_models(train_ml_flag=True, train_dl_flag=True):
    if train_ml_flag:
        train_ml()
    if train_dl_flag:
        train_dl()

if __name__ == "__main__":
    train_all_models()
