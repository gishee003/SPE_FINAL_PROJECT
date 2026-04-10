from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from scipy.stats import ks_2samp
import requests
import os

app = Flask(__name__)

# Path to PVC mount
PVC_PATH = "/data/churn-model"

# Load reference distributions
reference_file = os.path.join(PVC_PATH, "reference_distribution.pkl")
if os.path.exists(reference_file):
    with open(reference_file, "rb") as f:
        reference = pickle.load(f)
else:
    reference = None

@app.route('/drift', methods=['POST'])
def detect_drift():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        if reference is None:
            return jsonify({"drift_detected": False, "message": "No reference distribution found"}), 200

        drift_report = {}

        # 1. Data Drift: KS test for each numeric feature
        for feature in reference["feature_means"].keys():
            if feature in df.columns:
                stat, p_value = ks_2samp(
                    df[feature].values,
                    np.random.normal(
                        reference["feature_means"][feature],
                        reference["feature_stds"][feature],
                        len(df)
                    )
                )
                drift_report[f"{feature}_drift"] = bool(p_value < 0.05)

        # 2. Label Drift: compare Exited distribution
        if "Exited" in df.columns:
            new_dist = df["Exited"].value_counts(normalize=True).to_dict()
            drift_report["label_drift"] = bool(any(
                abs(new_dist.get(k, 0) - reference["label_distribution"].get(k, 0)) > 0.1
                for k in reference["label_distribution"].keys()
            ))


        # 3. Concept Drift: placeholder
        drift_report["concept_drift"] = "monitor prediction accuracy over time"

        drift_detected = any(v is True for v in drift_report.values())

        response = {"drift_detected": drift_detected, "details": drift_report}

        # Trigger retraining if drift detected
        if drift_detected:
            training_url = os.getenv("TRAINING_URL")
            if training_url:
                train_response = requests.post(training_url, json=data)
                response["training_response"] = train_response.json()

        return jsonify(response)

    except Exception as e:
        print("Drift detection error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
