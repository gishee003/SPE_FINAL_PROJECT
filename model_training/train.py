from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import time
import uuid
import threading
from datetime import datetime, timezone
import psutil

app = Flask(__name__)

# Path where PVC is mounted
data_path = "/data/churn-model/train.csv"
pvc_path = "/data/churn-model"

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

@app.route('/train', methods=['POST'])
def train_model():
    start_perf = time.perf_counter()
    training_run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    stop_event = threading.Event()
    samples = []
    sampler_thread = None

    # Training start event (useful for timelines).
    log_event(
        "train",
        "success",
        extra={"training_run_id": training_run_id, "training_started_at": started_at},
        event_type="training_run_started",
    )

    try:
        if request.is_json:
            payload = request.get_json(silent=True)
            if payload and isinstance(payload, list):
                df = pd.DataFrame(payload)
            else:
                # Empty or invalid JSON → fall back to CSV
                df = pd.read_csv(data_path)
        else:
            # No JSON header → fall back to CSV
            df = pd.read_csv(data_path)

        required_cols = {"Exited", "Geography", "Gender"}
        if not required_cols.issubset(df.columns):
            log_event(
                "train",
                "error",
                extra={
                    "training_run_id": training_run_id,
                    "error": {
                        "type": "validation_error",
                        "code": "missing_required_columns",
                        "message": "Dataset missing required columns",
                    },
                },
                event_type="training_run_failed",
            )
            return jsonify({"error": "Dataset missing required columns"}), 500

        # Features and target
        X = df.drop(columns=["Exited", "Surname", "id"], errors="ignore")
        y = df["Exited"]

        # Identify categorical and numeric columns
        categorical = ["Geography", "Gender"]
        numeric = [col for col in X.columns if col not in categorical]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", StandardScaler(), numeric),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

        # Start resource sampling before training begins.
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)  # initialize CPU percent

        def sample_resources():
            while not stop_event.is_set():
                cpu_percent = proc.cpu_percent(interval=None)
                rss_bytes = proc.memory_info().rss
                elapsed_ms = int((time.perf_counter() - start_perf) * 1000)
                samples.append((elapsed_ms, cpu_percent, rss_bytes))
                log_event(
                    "train",
                    "success",
                    extra={
                        "training_run_id": training_run_id,
                        "training_elapsed_ms": elapsed_ms,
                        "resource": {"cpu_percent": cpu_percent, "rss_bytes": rss_bytes},
                    },
                    event_type="resource_sample",
                )
                time.sleep(1)

        sampler_thread = threading.Thread(target=sample_resources, daemon=True)
        sampler_thread.start()

        # Train/validate split for accuracy trends.
        # For very small datasets, stratified splitting can fail.
        can_stratify = (
            (y.nunique() > 1)
            and (y.value_counts().min() >= 2)
            and (len(X) >= 5)
        )
        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        validation_accuracy = float(accuracy_score(y_val, y_pred))

        stop_event.set()
        if sampler_thread:
            sampler_thread.join(timeout=5)

        # Save model to PVC (pipeline includes preprocessing!)
        model_path = os.path.join(pvc_path, "churn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save reference distributions to PVC
        reference = {
            "feature_means": X[numeric].mean().to_dict(),
            "feature_stds": X[numeric].std().to_dict(),
            "label_distribution": y.value_counts(normalize=True).to_dict(),
        }
        ref_path = os.path.join(pvc_path, "reference_distribution.pkl")
        with open(ref_path, "wb") as f:
            pickle.dump(reference, f)

        completed_at = datetime.now(timezone.utc).isoformat()
        training_duration_ms = int((time.perf_counter() - start_perf) * 1000)

        # Compute summary stats from samples.
        if samples:
            max_rss_bytes = max(s[2] for s in samples)
            avg_rss_bytes = sum(s[2] for s in samples) / len(samples)
            max_cpu_percent = max(s[1] for s in samples)
            avg_cpu_percent = sum(s[1] for s in samples) / len(samples)
        else:
            max_rss_bytes = None
            avg_rss_bytes = None
            max_cpu_percent = None
            avg_cpu_percent = None

        log_event(
            "train",
            "success",
            extra={
                "training_run_id": training_run_id,
                "training_completed_at": completed_at,
                # `duration_ms` matches other services + Kibana duration panels.
                "duration_ms": training_duration_ms,
                "training_duration_ms": training_duration_ms,
                "validation_accuracy": validation_accuracy,
                "model_version": "v1.0",
                "resource_summary": {
                    "max_rss_bytes": max_rss_bytes,
                    "avg_rss_bytes": avg_rss_bytes,
                    "max_cpu_percent": max_cpu_percent,
                    "avg_cpu_percent": avg_cpu_percent,
                },
            },
            event_type="training_run_completed",
        )

        return jsonify({"status": "success", "message": "Model and reference saved to PVC"})

    except Exception as e:
        stop_event.set()
        if sampler_thread:
            try:
                sampler_thread.join(timeout=5)
            except Exception:
                pass

        completed_at = datetime.now(timezone.utc).isoformat()
        log_event(
            "train",
            "error",
            extra={
                "training_run_id": training_run_id,
                "training_completed_at": completed_at,
                "error": {
                    "type": "training_error",
                    "code": "training_exception",
                    "message": str(e),
                },
            },
            event_type="training_run_failed",
        )
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # threaded=True so /health stays responsive during long synchronous POST /train (Kubernetes probes).
    app.run(host='0.0.0.0', port=5001, threaded=True)
