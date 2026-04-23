import pickle

REF_PATH = "/home/kirti/SPE/SPE_FINAL_PROJECT/data/churn-model/reference_distribution.pkl"

with open(REF_PATH, "rb") as f:
    ref = pickle.load(f)

print(f"Feature means keys: {list(ref['feature_means'].keys())}")
print(f"Count: {len(ref['feature_means'])}")
