from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # Only keep training features
        X = df[["Age", "Tenure", "Balance"]]

        predictions = model.predict(X)
        results = ["Yes" if p == 1 else "No" for p in predictions]

        # Optionally return customerID alongside predictions
        if "customerID" in df.columns:
            output = [{"customerID": cid, "prediction": res}
                      for cid, res in zip(df["customerID"], results)]
            return jsonify({"predictions": output})
        else:
            return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
