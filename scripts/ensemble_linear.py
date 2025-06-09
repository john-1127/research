import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from cuml.ensemble import RandomForestRegressor

def train_linear_ensemble(csv_A, csv_B, csv_true, output_weights_path="./weights.csv"):
    df_A = pd.read_csv(csv_A)
    df_B = pd.read_csv(csv_B)
    df_Y = pd.read_csv(csv_true)

    smiles = df_A.iloc[:, 0]
    A = df_A.iloc[:, 1:].values
    B = df_B.iloc[:, 1:].values
    Y = df_Y.iloc[:, 1:].values


    num_dims = A.shape[1]
    weights = []

    for i in range(num_dims):
        X = np.stack([A[:, i], B[:, i]], axis=1)
        y = Y[:, i]
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        weights.append(reg.coef_)

    weights = np.array(weights)
    pd.DataFrame(weights, columns=["w_A", "w_B"]).to_csv(output_weights_path, index=False)
    return weights


def apply_ensemble_weights(csv_A, csv_B, weights_csv, output_csv="./ensemble.csv"):
    df_A = pd.read_csv(csv_A)
    df_B = pd.read_csv(csv_B)
    smiles = df_A.iloc[:, 0]

    A = df_A.iloc[:, 1:].values
    B = df_B.iloc[:, 1:].values
    W = pd.read_csv(weights_csv).values


    final = A * W[:, 0] + B * W[:, 1]

    df_final = pd.DataFrame(final)
    df_final.insert(0, "SMILES", smiles)
    df_final.to_csv(output_csv, index=False)


def train_rf_ensemble(csv_A, csv_B, csv_true, output_weights_path="./rf_weights.csv"):
    import joblib
    df_A = pd.read_csv(csv_A)
    df_B = pd.read_csv(csv_B)
    df_Y = pd.read_csv(csv_true)

    smiles = df_A.iloc[:, 0]
    A = df_A.iloc[:, 1:].values
    B = df_B.iloc[:, 1:].values
    Y = df_Y.iloc[:, 1:].values

    num_dims = A.shape[1]
    weights = []

    for i in range(num_dims):
        X = np.stack([A[:, i], B[:, i]], axis=1)
        y = Y[:, i]
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_stream=1)
        reg.fit(X, y)
        weights.append(reg)

    joblib.dump(weights, output_weights_path)

    return weights

def apply_rf_ensemble(csv_A, csv_B, model_path, output_csv="./ensemble_rf.csv"):
    import joblib
    df_A = pd.read_csv(csv_A)
    df_B = pd.read_csv(csv_B)
    smiles = df_A.iloc[:, 0]
    A = df_A.iloc[:, 1:].values
    B = df_B.iloc[:, 1:].values

    rf_models = joblib.load(model_path)
    num_dims = A.shape[1]
    final = []

    for i in range(num_dims):
        X = np.stack([A[:, i], B[:, i]], axis=1)
        y_pred = rf_models[i].predict(X)
        final.append(y_pred)

    final = np.stack(final, axis=1)  # shape: (n_samples, num_dims)
    df_final = pd.DataFrame(final)
    df_final.insert(0, "SMILES", smiles)
    df_final.to_csv(output_csv, index=False)


import time
start_time = time.time()

classical_file = "./output/model/classical_2100_layer3/fold_0/classical_2100_layer3.csv"
hybrid_file = "./output/model/qh2_2100_layer3/fold_0/qh2_2100_layer3.csv"
target_file = "./data/research_data/train_full.csv"

# train_rf_ensemble(classical_file, hybrid_file, target_file)
apply_rf_ensemble(classical_file, hybrid_file, "./rf_weights.csv")
end_time = time.time()
print("time:",end_time - start_time)

# apply_ensemble_weights(classical_file, hybrid_file, "./weights.csv")
