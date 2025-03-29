import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

def apply_ensemble_weights(csv_A, csv_B, weights_csv, output_csv="./final_pred.csv"):
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

classical_file = "../output/model/morgan_classical/fold_0/classical.csv"
hybrid_file = "../output/model/morgan_hybrid_probs/fold_0/morgan_hybrid_probs.csv"
target_file = "../data/research_data/test_full.csv"

train_linear_ensemble(classical_file, hybrid_file, target_file)
apply_ensemble_weights(classical_file, hybrid_file, "./weights.csv")
