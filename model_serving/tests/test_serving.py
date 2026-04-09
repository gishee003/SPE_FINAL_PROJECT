import unittest
from unittest.mock import patch, MagicMock
import os

# Adjust this import to match your folder/file name
from model_serving.serve import app 

class TestModelServing(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('model_serving.serve.model')  # Mock the global 'model' object
    def test_predict_success(self, mock_model):
        # 1. Setup Mock Model behavior
        mock_model.predict.return_value = [1, 0]
        mock_model.predict_proba.return_value = [[0.2, 0.8], [0.9, 0.1]]

        # 2. Define valid Bank Churn payload
        payload = [
            {
                "CustomerId": 12345,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0
            },
            {
                "CustomerId": 67890,
                "CreditScore": 700,
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 35,
                "Tenure": 5,
                "Balance": 80000.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 60000.0
            }
        ]

        # 3. Make Request
        response = self.app.post('/predict', json=payload)
        data = response.get_json()

        # 4. Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["prediction"], 1)
        self.assertEqual(data["results"][1]["prediction"], 0)
        self.assertIn("churn_probability", data["results"][0])

        # Verify that the model's predict method was called
        mock_model.predict.assert_called_once()

    def test_no_model_error(self):
        # Simulate the scenario where the model failed to load (model is None)
        with patch('model_serving.serve.model', None):
            payload = [{
                "CustomerId": 12345,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0
            }]
            response = self.app.post('/predict', json=payload)
            data = response.get_json()

            self.assertEqual(response.status_code, 500)
            self.assertIn("error", data)

if __name__ == '__main__':
    unittest.main()
