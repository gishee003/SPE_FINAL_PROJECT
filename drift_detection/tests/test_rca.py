import unittest
import pandas as pd
import numpy as np
import pickle
import os
import json
from unittest.mock import patch, MagicMock
from drift_detection.rca_xai import DriftRCAExplainer, generate_rca_report

class DummyModel:
    def predict_proba(self, X):
        # Return dummy probabilities for 2 classes
        return np.array([[0.5, 0.5]] * len(X))
    def predict(self, X):
        return np.array([0] * len(X))

class TestRCA(unittest.TestCase):
    def setUp(self):
        self.model_path = "/tmp/test_model.pkl"
        self.train_csv = "/tmp/test_train.csv"
        self.test_csv = "/tmp/test_test.csv"
        
        with open(self.model_path, "wb") as f:
            pickle.dump(DummyModel(), f)
            
        # Create dummy data
        train_data = pd.DataFrame({
            "Age": [20, 30, 40, 50],
            "Balance": [1000, 2000, 3000, 4000],
            "Exited": [0, 0, 1, 1],
            "CustomerId": [1, 2, 3, 4],
            "Surname": ["A", "B", "C", "D"]
        })
        train_data.to_csv(self.train_csv, index=False)
        
        test_data = pd.DataFrame({
            "Age": [60, 70],
            "Balance": [10000, 20000],
            "Exited": [1, 1],
            "CustomerId": [5, 6],
            "Surname": ["E", "F"]
        })
        test_data.to_csv(self.test_csv, index=False)

    def tearDown(self):
        for f in [self.model_path, self.train_csv, self.test_csv]:
            if os.path.exists(f):
                os.remove(f)

    @patch("shap.Explainer")
    def test_rca_logic(self, mock_explainer):
        # Mock SHAP explainer
        mock_instance = MagicMock()
        # Mock SHAP values (values, base_values, data)
        # For 2 features (Age, Balance) and 2 rows in test
        mock_shap_values = MagicMock()
        mock_shap_values.values = np.random.rand(2, 2, 2) # [rows, features, classes]
        mock_instance.return_value = mock_shap_values
        mock_explainer.return_value = mock_instance
        
        report = generate_rca_report(
            model_path=self.model_path,
            train_csv_path=self.train_csv,
            test_csv_path=self.test_csv,
            drifted_features=["Age"]
        )
        
        self.assertEqual(report["report_type"], "drift_root_cause_analysis")
        self.assertIn("Age", report["drifted_features"])
        self.assertIn("plain_english_explanation", report)
        # Check that IDs were dropped
        self.assertNotIn("CustomerId", report["baseline_feature_importance"])
        self.assertNotIn("Surname", report["baseline_feature_importance"])

    def test_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            explainer = DriftRCAExplainer(
                model_path=self.model_path,
                train_csv_path="non_existent.csv",
                test_csv_path=self.test_csv
            )
            explainer.load()

if __name__ == "__main__":
    unittest.main()
