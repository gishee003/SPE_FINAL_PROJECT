import json
import os
import tempfile
import unittest
import model_training.train as training_app

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.client = training_app.app.test_client()

    def test_train_model_success(self):
        tmpdir = tempfile.mkdtemp()
        training_app.pvc_path = tmpdir

        # Minimal valid Bank Churn payload
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
                "EstimatedSalary": 50000.0,
                "Exited": 0
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
                "EstimatedSalary": 60000.0,
                "Exited": 1
            }
        ]

        response = self.client.post("/train", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("Model and reference saved", data["message"])
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "churn_model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "reference_distribution.pkl")))

    def test_train_model_invalid_payload(self):
        # Missing Exited column
        payload = [
            {
                "CustomerId": 12345,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male"
            }
        ]
        response = self.client.post("/train", json=payload)
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn("error", data)

if __name__ == '__main__':
    unittest.main()
