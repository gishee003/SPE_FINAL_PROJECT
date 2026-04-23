from flask import Flask, jsonify, request, send_from_directory
import os
import time
import json
import logging
import sys

try:
    from drift_detection.rca_xai import generate_rca_report
except (ImportError, ModuleNotFoundError):
    from rca_xai import generate_rca_report

app = Flask(__name__)

@app.route("/dashboard")
def dashboard():
    return send_from_directory(".", "dashboard.html")

@app.route("/get-distribution/<feature>")
def get_distribution(feature):
    try:
        try:
            from drift_detection.rca_xai import DriftRCAExplainer
        except:
            from rca_xai import DriftRCAExplainer
        
        model_path = os.getenv("MODEL_PATH", "/data/churn-model/churn_model.pkl")
        train_csv_path = os.getenv("TRAIN_CSV_PATH", "/data/churn-model/train.csv")
        test_csv_path = os.getenv("TEST_CSV_PATH", "/data/churn-model/test.csv")

        explainer = DriftRCAExplainer(
            model_path=model_path,
            train_csv_path=train_csv_path,
            test_csv_path=test_csv_path
        )
        explainer.load()
        
        if feature not in explainer.train_df.columns:
            return jsonify({"error": f"Feature {feature} not found in training data"}), 404
        if feature not in explainer.test_df.columns:
            return jsonify({"error": f"Feature {feature} not found in production data"}), 404

        # Get training distribution
        train_vals = explainer.train_df[feature].dropna().tolist()
        # Get production distribution (from test.csv)
        prod_vals = explainer.test_df[feature].dropna().tolist()
        
        return jsonify({
            "feature": feature,
            "train": train_vals,
            "production": prod_vals
        })
    except Exception as e:
        log_event("rca", "error", extra={"error": {"message": str(e)}, "feature": feature})
        return jsonify({"error": str(e)}), 500


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/run-rca", methods=["POST"])
def run_rca():
    started_perf = time.perf_counter()
    try:
        payload = request.get_json(silent=True) or {}
        drifted_features = payload.get("drifted_features", [])
        if not drifted_features:
            log_event("rca", "error", extra={"error": {"message": "drifted_features is required"}})
            return jsonify({"error": "drifted_features is required"}), 400

        model_path = payload.get("model_path", os.getenv("MODEL_PATH", "/data/churn-model/churn_model.pkl"))
        train_csv_path = payload.get("train_csv_path", os.getenv("TRAIN_CSV_PATH", "/data/churn-model/train.csv"))
        
        # Check for file existence before proceeding
        if not os.path.exists(model_path):
            err_msg = f"Model artifact not found at {model_path}"
            log_event("rca", "error", extra={"error": {"message": err_msg, "path": model_path}})
            return jsonify({"error": err_msg}), 404
            
        log_event("rca", "success", extra={"rca": {"step": "explaining_drift"}})
        report = generate_rca_report(
            model_path=model_path,
            train_csv_path=train_csv_path,
            test_csv_path=payload.get(
                "test_csv_path", os.getenv("TEST_CSV_PATH", "/data/churn-model/test.csv")
            ),
            reference_path=payload.get(
                "reference_path", os.getenv("REFERENCE_PATH", "/data/churn-model/reference_distribution.pkl")
            ),
            drifted_features=drifted_features,
            drifted_batch_records=payload.get("drifted_batch_records"),
            shift_threshold_ratio=float(
                payload.get(
                    "shift_threshold_ratio",
                    os.getenv("RCA_SHIFT_THRESHOLD_RATIO", "1.5"),
                )
            ),
        )
        
        duration_ms = int((time.perf_counter() - started_perf) * 1000)
        log_event(
            "rca", 
            "success", 
            extra={
                "duration_ms": duration_ms,
                "rca": {
                    "drifted_features_count": len(drifted_features),
                    "rogue_features_count": len(report.get("rogue_features", []))
                }
            }
        )
        return jsonify(report), 200
    except Exception as e:
        duration_ms = int((time.perf_counter() - started_perf) * 1000)
        log_event("rca", "error", extra={"error": {"message": str(e)}, "duration_ms": duration_ms})
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, threaded=True)
