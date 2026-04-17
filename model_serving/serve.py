from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

import time
import logging, json

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'   # ONLY print message (no INFO:root prefix)
)

def log_event(service, status, extra=None, event_type="request"):
    # Emit ECS-compatible fields to avoid collisions with reserved mappings.
    event = {
        "service": {"name": service},
        "event": {
            "outcome": "success" if status == "success" else "failure",
            "type": event_type,
        },
        "app": {"status": status},
    }
    if extra:
        # Deep-merge only the nested `event` object if provided.
        for k, v in extra.items():
            if k == "event" and isinstance(v, dict):
                event["event"].update(v)
            else:
                event[k] = v
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
    request_started_perf = time.perf_counter()
    try:
        if model is None:
            raise ValueError("Model not loaded")

        data = request.get_json()

        if isinstance(data, dict):
            df = pd.DataFrame([dict(data)])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            log_event(
                "predict",
                "error",
                extra={
                    "error": {
                        "type": "invalid_input_format",
                        "code": "invalid_request_body",
                        "message": "Invalid input format",
                    },
                    "http": {"request": {"method": request.method, "path": request.path}},
                },
                event_type="request",
            )
            return jsonify({"error": "Invalid input format"}), 400

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

        pred_values = preds.tolist() if hasattr(preds, "tolist") else preds
        prob_values = probs.tolist() if hasattr(probs, "tolist") else probs

        duration_ms = int((time.perf_counter() - request_started_perf) * 1000)
        client_source = (
            request.headers.get("X-Client-Source")
            or request.headers.get("x-client-source")
            or "unknown"
        )

        log_event(
            "predict",
            "success",
            extra={
                "customer_id": customer_id,
                "duration_ms": duration_ms,
                "client": {"source": client_source},
                "http": {"request": {"method": request.method, "path": request.path}},
                "prediction": {
                    "prediction": [int(x) for x in pred_values],
                    "churn_probability": [float(x) for x in prob_values],
                },
            },
            event_type="request",
        )

        return jsonify({"results": results, "status": "success"})
    
    except Exception as e:
        duration_ms = int((time.perf_counter() - request_started_perf) * 1000)

        err_type = "serving_exception"
        err_code = "unhandled_exception"
        err_message = str(e)

        if "Model not loaded" in err_message:
            err_type = "model_not_loaded"
            err_code = "model_missing"
        elif "Missing required feature" in err_message:
            err_type = "validation_error"
            err_code = "missing_required_feature"

        log_event(
            "predict",
            "error",
            extra={
                "error": {"type": err_type, "code": err_code, "message": err_message},
                "duration_ms": duration_ms,
                "http": {"request": {"method": request.method, "path": request.path}},
            },
            event_type="request",
        )
        return jsonify({"error": err_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
