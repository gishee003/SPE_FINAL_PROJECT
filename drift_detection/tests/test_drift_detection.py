import unittest
from unittest.mock import patch, MagicMock
import os
import json

# Adjust import to match your file structure
from drift_detection.drift_detection import app 

class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        os.environ["TRAINING_URL"] = "http://fake-training-service/train"

    @patch('drift_detection.drift_detection.reference')  # Mock the global 'reference' object
    @patch('requests.post')  # Mock the retraining trigger
    def test_detect_drift_success(self, mock_post, mock_reference):
        # 1. Setup Mock Reference Data
        mock_reference.__getitem__.side_effect = {
            "feature_means": {"Age": 40.0},
            "feature_stds": {"Age": 5.0},
            "label_distribution": {1: 0.2, 0: 0.8}
        }.get

        # Mock retraining response
        mock_train_resp = MagicMock()
        mock_train_resp.json.return_value = {"status": "retraining_started"}
        mock_post.return_value = mock_train_resp

        # 2. Define Payload (Data significantly different to trigger drift)
        # Age mean here is ~100, while reference is 40
        payload = [
            {
                "CustomerId": 1,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 100,
                "Tenure": 3,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 40000.0,
                "Exited": 1
            },
            {
                "CustomerId": 2,
                "CreditScore": 650,
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 110,
                "Tenure": 5,
                "Balance": 60000.0,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 45000.0,
                "Exited": 1
            }
        ]

        # 3. Make Request
        response = self.app.post('/drift', json=payload)
        data = response.get_json()

        # 4. Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("drift_detected", data)
        self.assertTrue(data["drift_detected"])  # Expect drift due to large Age difference

    def test_no_reference_found(self):
        # Patch reference to be None to test the 'No reference distribution found' logic
        with patch('drift_detection.drift_detection.reference', None):
            payload = [{
                "CustomerId": 1,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 40000.0,
                "Exited": 0
            }]
            response = self.app.post('/drift', json=payload)
            data = response.get_json()
            
            self.assertEqual(response.status_code, 200)
            self.assertFalse(data["drift_detected"])

if __name__ == '__main__':
    unittest.main()
