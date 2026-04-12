from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

app = Flask(__name__)

# Path where PVC is mounted
data_path = "/data/churn-model/train.csv"
pvc_path = "/data/churn-model"

import logging, json

logging.basicConfig(level=logging.INFO)

def log_event(service, status, extra=None):
    event = {"service": service, "status": status}
    if extra:
        event.update(extra)
    logging.info(json.dumps(event))

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # print("Listing /data/churn-model:", os.listdir("/data/churn-model"))

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
            return jsonify({"error": "Dataset missing required columns"}), 500

        # Features and target
        X = df.drop(columns=['Exited', 'Surname', 'id'], errors='ignore')  # drop non-numeric string columns
        y = df['Exited']

        # Identify categorical and numeric columns
        categorical = ['Geography', 'Gender']
        numeric = [col for col in X.columns if col not in categorical]


        # Preprocessing: one-hot encode categorical, scale numeric
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
                ('num', StandardScaler(), numeric)
            ]
        )

        # Build pipeline with preprocessing + model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        # Train model
        model.fit(X, y)

        # Save model to PVC (pipeline includes preprocessing!)
        model_path = os.path.join(pvc_path, "churn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save reference distributions to PVC
        reference = {
            "feature_means": X[numeric].mean().to_dict(),
            "feature_stds": X[numeric].std().to_dict(),
            "label_distribution": y.value_counts(normalize=True).to_dict()
        }
        ref_path = os.path.join(pvc_path, "reference_distribution.pkl")
        with open(ref_path, "wb") as f:
            pickle.dump(reference, f)
        
        print("Saved model to", model_path)
        print("Saved reference distribution to", ref_path)

        log_event("train", "success", {"model_version": "v1.0"})
        return jsonify({"status": "success", "message": "Model and reference saved to PVC"})
    
    except Exception as e:
        print("Model Training error:", e)
        log_event("train", "error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
