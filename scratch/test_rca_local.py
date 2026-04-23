import os
import sys
import pandas as pd

# Add the project root to sys.path
sys.path.append("/home/kirti/SPE/SPE_FINAL_PROJECT")

try:
    from drift_detection.rca_xai import generate_rca_report
except ImportError:
    from drift_detection.drift_detection.rca_xai import generate_rca_report

MODEL_PATH = "/home/kirti/SPE/SPE_FINAL_PROJECT/data/churn-model/churn_model.pkl"
TRAIN_CSV = "/home/kirti/SPE/SPE_FINAL_PROJECT/data/churn-model/train.csv"
TEST_CSV = "/home/kirti/SPE/SPE_FINAL_PROJECT/data/churn-model/test.csv"
REFERENCE_PKL = "/home/kirti/SPE/SPE_FINAL_PROJECT/data/churn-model/reference_distribution.pkl"

def test_rca():
    print("Testing RCA logic locally...")
    try:
        report = generate_rca_report(
            model_path=MODEL_PATH,
            train_csv_path=TRAIN_CSV,
            test_csv_path=TEST_CSV,
            reference_path=REFERENCE_PKL,
            drifted_features=["Age", "Balance"]
        )
        print("RCA report generated successfully!")
        print("Rogue features:", report.get("rogue_features"))
    except Exception as e:
        print(f"RCA failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rca()
