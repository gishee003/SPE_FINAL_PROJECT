from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Path to model inside PVC
model_path = "/data/churn-model/churn_model.pkl"

# Load model at startup
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # Drop target if accidentally included
        if 'Exited' in df.columns:
            df = df.drop(columns=['Exited'])

        # Make predictions
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]  # probability of churn

        results = []
        for i in range(len(df)):
            results.append({
                "CustomerId": df.iloc[i].get("CustomerId", None),
                "prediction": int(preds[i]),
                "churn_probability": float(probs[i])
            })

        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
