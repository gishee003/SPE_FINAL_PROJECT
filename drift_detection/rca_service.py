from flask import Flask, jsonify, request
import os

from rca_xai import generate_rca_report

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/run-rca", methods=["POST"])
def run_rca():
    try:
        payload = request.get_json(silent=True) or {}
        drifted_features = payload.get("drifted_features", [])
        if not drifted_features:
            return jsonify({"error": "drifted_features is required"}), 400

        report = generate_rca_report(
            model_path=payload.get("model_path", os.getenv("MODEL_PATH", "/data/churn-model/churn_model.pkl")),
            train_csv_path=payload.get(
                "train_csv_path", os.getenv("TRAIN_CSV_PATH", "/data/churn-model/train.csv")
            ),
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
        return jsonify(report), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
