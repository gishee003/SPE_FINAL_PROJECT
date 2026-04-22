from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from scipy.stats import ks_2samp
import requests
import os
import time
try:
    from drift_detection.rca_xai import generate_rca_report
except (ImportError, ModuleNotFoundError):
    from rca_xai import generate_rca_report


app = Flask(__name__)

PVC_PATH = "/data/churn-model"

import logging, json

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'   # ONLY print message (no INFO:root prefix)
)

def log_event(service, status, extra=None, event_type="request"):
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

        drift_report["concept_drift"] = "monitor prediction accuracy over time"

        drift_detected = any(v is True for v in drift_report.values())
        response = {"drift_detected": drift_detected, "details": drift_report}

        if drift_detected:
            training_url = os.getenv("TRAINING_URL")
            if training_url:
                train_response = requests.post(training_url, json=data)
                response["training_response"] = train_response.json()

            rca_enabled = os.getenv("ENABLE_XAI_RCA", "true").lower() == "true"
            if rca_enabled:
                drifted_features = [
                    fe["feature"]
                    for fe in feature_events
                    if fe.get("feature") != "label_drift" and fe.get("drifted")
                ]
                if drifted_features:
                    try:
                        response["rca_report"] = generate_rca_report(
                            model_path=os.getenv("MODEL_PATH", "/data/churn-model/churn_model.pkl"),
                            train_csv_path=os.getenv("TRAIN_CSV_PATH", "/data/churn-model/train.csv"),
                            test_csv_path=os.getenv("TEST_CSV_PATH", "/data/churn-model/test.csv"),
                            reference_path=reference_file,
                            drifted_features=drifted_features,
                            drifted_batch_records=data if isinstance(data, list) else [data],
                            shift_threshold_ratio=float(
                                os.getenv("RCA_SHIFT_THRESHOLD_RATIO", "1.5")
                            ),
                        )
                    except Exception as rca_exc:
                        response["rca_report"] = {
                            "report_type": "drift_root_cause_analysis",
                            "error": str(rca_exc),
                            "plain_english_explanation": (
                                "Drift was detected, but RCA report generation failed. "
                                "Verify model and CSV artifact availability for SHAP analysis."
                            ),
                        }

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

        for fe in feature_events:
            extra = {
                "drift": fe,
                "duration_ms": total_duration_ms,
                "http": {"request": {"method": request.method, "path": request.path}},
            }
            pv = fe.get("p_value")
            extra["drift_p_value"] = float(pv) if pv is not None else None
            log_event(
                "drift",
                "success",
                extra=extra,
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
