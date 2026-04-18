import unittest
from unittest.mock import patch, MagicMock
from data_ingestion.app import app
import os

class FlaskIngestTestCase(unittest.TestCase):
    def setUp(self):
        os.environ["SERVING_URL"] = "http://fake-serving-url.com"
        os.environ["DRIFT_URL"] = "http://fake-drift-url.com"
        
        self.app = app.test_client()
        self.app.testing = True

    @patch('requests.post')
    def test_ingest_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

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

        response = self.app.post('/ingest', json=payload)
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'ingested')
        self.assertEqual(data['rows'], 2)
        self.assertEqual(mock_post.call_count, 2)

    def test_ingest_invalid_schema(self):
        payload = [{
            "CustomerId": 12345,
            "CreditScore": 600,
            "Geography": "France"
        }]
        
        response = self.app.post('/ingest', json=payload)
        data = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing columns", data['error'])

if __name__ == '__main__':
    unittest.main()
