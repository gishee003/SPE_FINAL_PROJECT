from flask import Flask, request, jsonify
import pandas as pd
import requests
import os

app = Flask(__name__)

import logging, json

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'   # ONLY print message (no INFO:root prefix)
)

def log_event(service, status, extra=None):
    event = {"service": service, "status": status}
    if extra:
        event.update(extra)
    logging.info(json.dumps(event))

@app.route('/ingest', methods=['POST'])
def ingest_data():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # Bank Churn dataset schema validation
        required_columns = [
            'CustomerId', 'CreditScore', 'Geography', 'Gender',
            'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited'
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        # Forward to serving for predictions
        serving_url = os.getenv("SERVING_URL")
        print(f"Attempting to reach Serving at: {serving_url}")
        serving_response = requests.post(serving_url, json=data, timeout=5)

        # Forward to drift detection
        drift_url = os.getenv("DRIFT_URL")
        drift_response = requests.post(drift_url, json=data)

        log_event("ingest", "success", {"rows": len(data)})

        return jsonify({
            "status": "ingested",
            "rows": len(df),
            "serving_response": serving_response.json(),
            "drift_response": drift_response.json()
        })
    except Exception as e:
        log_event("ingest", "error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
