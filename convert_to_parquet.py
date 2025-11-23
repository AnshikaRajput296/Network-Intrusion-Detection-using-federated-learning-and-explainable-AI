import os
import numpy as np
DIRS = {
    "FedAvg": "data/fed_avg",
    "FedProx": "data/fed_prox",
    "DecisionTree": "data/DecisionTree"
}
folder_path = DIRS["FedAvg"]
def load_numpy(path):
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading numpy file {path}: {e}")
    return None
shap_vals = None
for fname in ["shap_values.npy", "shap_values_global.npy"]:
    shap_path = os.path.join(folder_path, fname)
    if os.path.exists(shap_path):
        shap_vals = load_numpy(shap_path)
        print(shap_vals.shape)
        break
