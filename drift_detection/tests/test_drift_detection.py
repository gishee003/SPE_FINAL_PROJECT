import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import json

# Assuming your file is named drift_app.py
from drift_detection.drift_detection import app 

class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        os.environ["TRAINING_URL"] = "http://fake-training-service/train"

    @patch('drift_detection.drift_detection.reference') # Mock the global 'reference' object
    @patch('requests.post')            # Mock the retraining trigger
    def test_detect_drift_success(self, mock_post, mock_reference):
        # 1. Setup Mock Reference Data
        # We simulate a reference distribution for 'tenure'
        mock_reference.__getitem__.side_effect = {
            "feature_means": {"tenure": 20.0},
            "feature_stds": {"tenure": 5.0},
            "label_distribution": {1: 0.2, 0: 0.8}
        }.get
        
        # Mock retraining response
        mock_train_resp = MagicMock()
        mock_train_resp.json.return_value = {"status": "retraining_started"}
        mock_post.return_value = mock_train_resp

        # 2. Define Payload (Data that is significantly different to trigger drift)
        # Tenure mean here is 100, while reference is 20
        payload = [
            {"customerID": "1", "tenure": 100, "Churn": "Yes"},
            {"customerID": "2", "tenure": 110, "Churn": "Yes"}
        ]

        # 3. Make Request
        response = self.app.post('/drift', json=payload)
        data = response.get_json()

        # 4. Assertions
        self.assertEqual(response.status_code, 200)
        self.assertIn("drift_detected", data)
        # Since 100 is far from 20, drift_detected should likely be True
        
    def test_no_reference_found(self):
        # Patch reference to be None to test the 'No reference distribution found' logic
        with patch('drift_detection.drift_detection.reference', None):
            payload = [{"customerID": "1", "tenure": 20}]
            response = self.app.post('/drift', json=payload)
            data = response.get_json()
            
            self.assertEqual(response.status_code, 200)
            self.assertFalse(data["drift_detected"])

if __name__ == '__main__':
    unittest.main()