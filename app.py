from flask import Flask, request, jsonify
from flask_cors import CORS
from detect_url import predict_url
from train_models import train_all_models

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "🎯 Phishing Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        result = predict_url(url)
        return jsonify(result)
    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/train/ml", methods=["POST"])
def train_ml():
    try:
        print("📦 Starting ML model training...")
        train_all_models(train_ml_flag=True, train_dl_flag=False)
        print("✅ ML training completed!")
        return jsonify({"message": "✅ ML model trained successfully"})
    except Exception as e:
        print("❌ ML Training Error:", e)
        return jsonify({"error": f"ML training failed: {str(e)}"}), 500

@app.route("/train/dl", methods=["POST"])
def train_dl():
    try:
        print("⚙️ Starting DL model training...")
        train_all_models(train_ml_flag=False, train_dl_flag=True)
        print("✅ DL training completed!")
        return jsonify({"message": "✅ DL model trained successfully"})
    except Exception as e:
        print("❌ DL Training Error:", e)
        return jsonify({"error": f"DL training failed: {str(e)}"}), 500

@app.route("/train/all", methods=["POST"])
def train_all():
    try:
        print("🧠 Training all models...")
        train_all_models(train_ml_flag=True, train_dl_flag=True)
        print("✅ All models trained!")
        return jsonify({"message": "✅ All models trained successfully"})
    except Exception as e:
        print("❌ Training All Models Error:", e)
        return jsonify({"error": f"Training all models failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)


