import unittest
import json
import os
import tempfile
import pandas as pd
from unittest.mock import patch
import model_training.train as training_app

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.client = training_app.app.test_client()
        self.tmpdir = tempfile.mkdtemp()
        training_app.pvc_path = self.tmpdir

    @patch('model_training.train.pd.read_csv')
    def test_train_model_success(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "CustomerId": [12345, 67890],
            "CreditScore": [600, 700],
            "Geography": ["France", "Germany"],
            "Gender": ["Male", "Female"],
            "Age": [40, 35],
            "Tenure": [3, 5],
            "Balance": [60000.0, 80000.0],
            "NumOfProducts": [2, 1],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 0],
            "EstimatedSalary": [50000.0, 60000.0],
            "Exited": [0, 1]
        })
        mock_read_csv.return_value = mock_df

        response = self.client.post("/train", json=[])
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("Model and reference saved", data["message"])

        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "churn_model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "reference_distribution.pkl")))

    def test_train_model_invalid_payload(self):
        payload = [{
            "CustomerId": 12345,
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Male"
        }]
        response = self.client.post("/train", json=payload)
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

if __name__ == '__main__':
    unittest.main()
