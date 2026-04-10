import unittest
from unittest.mock import patch
from model_serving.serve import app   # import from the package
import numpy as np

class TestModelServing(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("model_serving.serve.model", None)   # patch inside the package
    def test_no_model_error(self):
        payload = [{
            "CustomerId": 15634602,
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        }]
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertIn("error", data)

    @patch("model_serving.serve.model")   # patch the correct module path
    def test_predict_success(self, mock_model):
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        payload = [{
            "CustomerId": 15634602,
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 0.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        }]
        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["results"][0]["prediction"], 1)
        self.assertAlmostEqual(data["results"][0]["churn_probability"], 0.8)

if __name__ == "__main__":
    unittest.main()
