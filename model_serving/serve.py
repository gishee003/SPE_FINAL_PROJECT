from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

import logging, json

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'   # ONLY print message (no INFO:root prefix)
)

def log_event(service, status, extra=None):
    # Emit ECS-compatible fields to avoid collisions with reserved mappings.
    event = {
        "service": {"name": service},
        "event": {"outcome": "success" if status == "success" else "failure"},
        "app": {"status": status}
    }
    if extra:
        event.update(extra)
    logging.info(json.dumps(event))


app = Flask(__name__)

# Path to model inside PVC
model_path = "/data/churn-model/churn_model.pkl"

# Load model at startup
model = None
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise ValueError("Model not loaded")

        data = request.get_json()

        logging.info("Received type: %s", type(data))
        logging.info("Received data: %s", data)
        print("DEBUG type:", type(data))
        print("DEBUG data:", data)

        if isinstance(data, dict):
            df = pd.DataFrame([dict(data)])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format"})

        # Drop target if accidentally included
        if 'Exited' in df.columns:
            df = df.drop(columns=['Exited'])

        # Ensure required Bank Churn features exist
        required_features = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary"
        ]
        for feat in required_features:
            if feat not in df.columns:
                raise ValueError(f"Missing required feature: {feat}")

        # Make predictions using the full pipeline (preprocessor + classifier)
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        results = []
        for i in range(len(df)):
            results.append({
                "CustomerId": int(df.iloc[i].get("CustomerId", 0)),
                "prediction": int(preds[i]),
                "churn_probability": float(probs[i])
            })
            
        if isinstance(data, dict):
            customer_id = data.get("CustomerId")
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            customer_id = data[0].get("CustomerId")
        else:
            customer_id = None

        log_event("predict", "success", {"customer_id": customer_id})

        return jsonify({"results": results, "status": "success"})
    
    except Exception as e:
        log_event("predict", "error", {"error": str(e)})
        print("Model Serving error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
