from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
import sys
import time

app = Flask(__name__)

import logging, json

# --- Structured Logging Setup ---
# We use a custom format to ensure ONLY the JSON string is printed
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
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
        for k, v in extra.items():
            if k == "event" and isinstance(v, dict):
                event["event"].update(v)
            else:
                event[k] = v
    logging.info(json.dumps(event))

@app.route('/ingest', methods=['POST'])
def ingest_data():
    try:
        started_perf = time.perf_counter()
        data = request.get_json()
        df = pd.DataFrame(data)

        ingest_source = (
            request.headers.get("X-Source")
            or request.headers.get("x-source")
            or "unknown"
        )
        if isinstance(data, dict):
            ingest_source = data.get("source", ingest_source)

        # Bank Churn dataset schema validation
        required_columns = [
            'CustomerId', 'CreditScore', 'Geography', 'Gender',
            'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            log_event(
                "ingest",
                "error",
                extra={
                    "error": {
                        "type": "validation_error",
                        "code": "missing_columns",
                        "message": "Missing required columns",
                        "missing_columns": missing,
                    },
                    "http": {"request": {"method": request.method, "path": request.path}},
                    "rows_requested": len(data) if isinstance(data, list) else 1,
                },
                event_type="request",
            )
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # Forward to serving for predictions
        serving_url = os.getenv("SERVING_URL")
        serving_started_perf = time.perf_counter()
        serving_response = requests.post(serving_url, json=data, timeout=5)
        serving_latency_ms = int((time.perf_counter() - serving_started_perf) * 1000)

        # Forward to drift detection
        drift_url = os.getenv("DRIFT_URL")
        drift_started_perf = time.perf_counter()
        drift_response = requests.post(drift_url, json=data)
        drift_latency_ms = int((time.perf_counter() - drift_started_perf) * 1000)

        total_latency_ms = int((time.perf_counter() - started_perf) * 1000)

        log_event(
            "ingest",
            "success",
            extra={
                "rows": len(data) if isinstance(data, list) else 1,
                "latency_ms": total_latency_ms,
                "ingest": {"source": ingest_source},
                "dependencies": {
                    "serving_latency_ms": serving_latency_ms,
                    "drift_latency_ms": drift_latency_ms,
                },
                "http": {"request": {"method": request.method, "path": request.path}},
            },
            event_type="request",
        )

        return jsonify({
            "status": "ingested",
            "rows": len(df),
            "serving_response": serving_response.json(),
            "drift_response": drift_response.json()
        })
    except Exception as e:
        log_event(
            "ingest",
            "error",
            extra={
                "error": {
                    "type": "ingestion_exception",
                    "code": "unhandled_exception",
                    "message": str(e),
                },
                "http": {"request": {"method": request.method, "path": request.path}},
            },
            event_type="request",
        )
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
