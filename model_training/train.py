from flask import Flask, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

app = Flask(__name__)

# Path where PVC is mounted
data_path = "/data/churn-model/train.csv"
pvc_path = "/data/churn-model"

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Load dataset directly from mounted PVC
        df = pd.read_csv(data_path)

        # Features and target
        X = df.drop(columns=['Exited'])
        y = df['Exited']  # Already 0/1 in Kaggle dataset

        # Identify categorical and numeric columns
        categorical = ['Geography', 'Gender']
        numeric = [col for col in X.columns if col not in categorical]

        # Preprocessing: one-hot encode categorical, passthrough numeric
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
                ('num', 'passthrough', numeric)
            ]
        )

        # Build pipeline with preprocessing + model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        # Train model
        model.fit(X, y)

        # Save model to PVC
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

        return jsonify({"status": "success", "message": "Model and reference saved to PVC"})
    
    except Exception as e:
        print("Model Training error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
