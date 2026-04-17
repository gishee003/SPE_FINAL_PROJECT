from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from scipy.stats import ks_2samp
import requests
import os
import time

app = Flask(__name__)

# Path to PVC mount
PVC_PATH = "/data/churn-model"

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
        for k, v in extra.items():
            if k == "event" and isinstance(v, dict):
                event["event"].update(v)
            else:
                event[k] = v
    logging.info(json.dumps(event))

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
        started_perf = time.perf_counter()
        data = request.get_json()
        df = pd.DataFrame(data)

        if reference is None:
            log_event(
                "drift",
                "success",
                extra={
                    "drift": {"drift_detected": False},
                    "duration_ms": int((time.perf_counter() - started_perf) * 1000),
                    "http": {"request": {"method": request.method, "path": request.path}},
                    "error": {
                        "type": "reference_not_found",
                        "code": "no_reference_distribution",
                        "message": "No reference distribution found",
                    },
                },
                event_type="drift_overall",
            )
            return (
                jsonify(
                    {"drift_detected": False, "message": "No reference distribution found"}
                ),
                200,
            )

        drift_report = {}
        feature_events = []

        # 1. Data Drift: KS test for each numeric feature
        for feature in reference["feature_means"].keys():
            if feature in df.columns:
                stat, p_value = ks_2samp(
                    df[feature].values,
                    np.random.normal(
                        reference["feature_means"][feature],
                        reference["feature_stds"][feature],
                        len(df),
                    ),
                )

                drifted = bool(p_value < 0.05)
                drift_report[f"{feature}_drift"] = drifted

                reference_mean = float(reference["feature_means"].get(feature, 0.0))
                reference_std = float(reference["feature_stds"].get(feature, 0.0))
                live_mean = float(df[feature].mean())
                live_std = float(df[feature].std())

                # Confidence: higher when p-value is smaller.
                confidence = float(max(0.0, min(1.0, 1.0 - p_value)))
                if p_value <= 0.01:
                    severity = "high"
                elif p_value <= 0.05:
                    severity = "medium"
                else:
                    severity = "low"

                feature_events.append(
                    {
                        "feature": feature,
                        "drifted": drifted,
                        "p_value": float(p_value),
                        "confidence": confidence,
                        "severity": severity,
                        "reference_mean": reference_mean,
                        "reference_std": reference_std,
                        "live_mean": live_mean,
                        "live_std": live_std,
                    }
                )

        # 2. Label Drift: compare Exited distribution
        if "Exited" in df.columns:
            new_dist = df["Exited"].value_counts(normalize=True).to_dict()
            max_abs_diff = max(
                abs(new_dist.get(k, 0) - reference["label_distribution"].get(k, 0))
                for k in reference["label_distribution"].keys()
            )
            label_drifted = bool(max_abs_diff > 0.1)
            drift_report["label_drift"] = label_drifted

            confidence = float(max(0.0, min(1.0, max_abs_diff)))
            severity = (
                "high"
                if max_abs_diff >= 0.2
                else "medium"
                if max_abs_diff >= 0.1
                else "low"
            )
            feature_events.append(
                {
                    "feature": "label_drift",
                    "drifted": label_drifted,
                    "p_value": None,
                    "confidence": confidence,
                    "severity": severity,
                    "reference_mean": None,
                    "reference_std": None,
                    "live_mean": None,
                    "live_std": None,
                }
            )

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

        total_duration_ms = int((time.perf_counter() - started_perf) * 1000)

        log_event(
            "drift",
            "success",
            extra={
                "drift": {
                    "drift_detected": drift_detected,
                    "feature_drifted_count": sum(1 for fe in feature_events if fe.get("drifted")),
                },
                "duration_ms": total_duration_ms,
                "http": {"request": {"method": request.method, "path": request.path}},
            },
            event_type="drift_overall",
        )

        # Emit one log event per feature drift (used for charts).
        for fe in feature_events:
            log_event(
                "drift",
                "success",
                extra={
                    "drift": fe,
                    "duration_ms": total_duration_ms,
                    "http": {"request": {"method": request.method, "path": request.path}},
                },
                event_type="feature_drift",
            )

        return jsonify(response)

    except Exception as e:
        log_event(
            "drift",
            "error",
            extra={
                "error": {
                    "type": "drift_exception",
                    "code": "unhandled_exception",
                    "message": str(e),
                },
                "http": {"request": {"method": request.method, "path": request.path}},
                "duration_ms": int((time.perf_counter() - started_perf) * 1000)
                if "started_perf" in locals()
                else None,
            },
            event_type="drift_error",
        )
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
