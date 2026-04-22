import unittest
from unittest.mock import patch, MagicMock
import os
import json

import drift_detection.drift_detection as drift_detection 

class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        self.app = drift_detection.app.test_client()
        self.app.testing = True
        os.environ["TRAINING_URL"] = "http://fake-training-service/train"

        drift_detection.reference = {
            "feature_means": {
                "Age": 40.0,
                "CreditScore": 650.0,
                "Balance": 50000.0,
                "Tenure": 5.0,
                "EstimatedSalary": 45000.0,
                "NumOfProducts": 1.0,
                "HasCrCard": 0.5,
                "IsActiveMember": 0.5
            },
            "feature_stds": {
                "Age": 5.0,
                "CreditScore": 50.0,
                "Balance": 10000.0,
                "Tenure": 2.0,
                "EstimatedSalary": 5000.0,
                "NumOfProducts": 0.5,
                "HasCrCard": 0.5,
                "IsActiveMember": 0.5
            },
            "label_distribution": {1: 0.2, 0: 0.8}
        }
        
    @patch('drift_detection.drift_detection.generate_rca_report')
    @patch('requests.post')
    def test_detect_drift_success(self, mock_post, mock_rca):
        mock_train_resp = MagicMock()
        mock_train_resp.json.return_value = {"status": "retraining_started"}
        mock_post.return_value = mock_train_resp
        
        mock_rca.return_value = {
            "report_type": "drift_root_cause_analysis",
            "drifted_features": ["Age"],
            "rogue_features": ["Age"],
            "plain_english_explanation": "Test RCA"
        }

        payload = [
            {
                "CustomerId": i,
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 100 + i, # Highly drifted
                "Tenure": 3,
                "Balance": 90000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 40000.0,
                "Exited": 1
            } for i in range(10)
        ]

        response = self.app.post('/drift', json=payload)
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("drift_detected", data)
        self.assertTrue(data["drift_detected"])  # Expect drift due to large Age/Balance difference
        self.assertIn("rca_report", data)
        self.assertEqual(data["rca_report"]["report_type"], "drift_root_cause_analysis")

    def test_no_reference_found(self):
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
