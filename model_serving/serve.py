from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Path to model inside PVC
model_path = "/data/churn-model/churn_model.pkl"

# Load model at startup
model = None
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise ValueError("Model not loaded")

        data = request.get_json()
        df = pd.DataFrame(data)

        # Drop target if accidentally included
        if 'Exited' in df.columns:
            df = df.drop(columns=['Exited'])

        # Ensure required Bank Churn features exist
        required_features = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary"
        ]
        for feat in required_features:
            if feat not in df.columns:
                raise ValueError(f"Missing required feature: {feat}")

        # Make predictions
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]  # probability of churn

        results = []
        for i in range(len(df)):
            results.append({
                "CustomerId": int(df.iloc[i].get("CustomerId", 0)),  # cast to int
                "prediction": int(preds[i]),                         # cast to int
                "churn_probability": float(probs[i])                 # cast to float
            })

        return jsonify({"status": "success", "results": results})
    
    except Exception as e:
        print("Model Serving error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
