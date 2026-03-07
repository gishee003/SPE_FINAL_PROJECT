from flask import Flask, request, jsonify
import pandas as pd
import requests   # <-- add this

app = Flask(__name__)

@app.route('/ingest', methods=['POST'])
def ingest_data():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        if 'customerID' not in df.columns or 'Churn' not in df.columns:
            return jsonify({"error": "Invalid schema"}), 400

        # Forward cleaned data to training microservice
        response = requests.post(
            "http://model-training-service:5001/train",  # service name in Kubernetes
            json=data
        )

        return jsonify({
            "status": "forwarded to training",
            "rows": len(df),
            "training_response": response.json()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
