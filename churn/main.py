import xgboost as xgb
from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd

# Paths to Model & Scaler
MODEL_PATH = "customer_churn_model.json"
SCALER_PATH = "scaler.pkl"

# Load Model & Scaler
loaded_model = xgb.Booster()
loaded_model.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Define feature names (must match training data)
FEATURE_NAMES = [
    "Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Subscription Type", "Contract Length",
    "Total Spend", "Last Interaction", "Engagement Score"
]

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    """ Home route with instructions """
    return jsonify({
        "message": "Welcome to the Churn Prediction API!",
        "Flask API URL": "http://127.0.0.1:5000/",
        "Streamlit UI URL": "http://127.0.0.1:8501/",
        "Instructions": "Send a POST request to /predict with customer data to get churn predictions."
    })

@app.route("/predict", methods=["POST"])
def predict():
    """ Churn prediction endpoint """
    try:
        # Get JSON Data
        data = request.get_json()

        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request JSON"}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame(data["features"])

        # Check if all required features are present
        missing_columns = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

        # Ensure correct column order
        df = df[FEATURE_NAMES]

        # Scale Input Data
        input_scaled = loaded_scaler.transform(df)

        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(input_scaled, feature_names=FEATURE_NAMES)

        # Make Predictions
        predictions = loaded_model.predict(dmatrix)

        return jsonify({"churn_predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n✅ Flask API running at: http://127.0.0.1:5000/")
    print("✅ Streamlit UI running at: http://127.0.0.1:8501/\n")
    app.run(host="0.0.0.0", port=5000)
