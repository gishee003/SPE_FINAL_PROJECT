import pickle
import os

MODEL_PATH = "/home/kirti/SPE/SPE_FINAL_PROJECT/data/churn-model/churn_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
if hasattr(model, "feature_names_in_"):
    print(f"Feature names in: {model.feature_names_in_}")
    print(f"Count: {len(model.feature_names_in_)}")
elif hasattr(model, "named_steps"):
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor:
        print("Preprocessor found.")
        # Some sklearn versions have this
        try:
            print(f"Feature names in: {preprocessor.feature_names_in_}")
            print(f"Count: {len(preprocessor.feature_names_in_)}")
        except:
            print("Could not find feature_names_in_")
