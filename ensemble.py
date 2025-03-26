import pandas as pd

df1 = pd.read_csv("./output/model/morgan_classical/fold_0/classical.csv")
df2 = pd.read_csv("./output/model/morgan_hybrid_probs/fold_0/morgan_hybrid_probs.csv")

def ensemble(df1, df2, weight1, weight2):

    columns_to_process = [col for col in df1.columns if col.isdigit() and 400 <= int(col) <= 4000]

    df_avg = df1.copy()
    df_avg[columns_to_process] = (df1[columns_to_process] * 8+ df2[columns_to_process] * 2) / 10

    df_avg.to_csv("./test_ensemble_weight_2_3.csv", index=False)


def ensemble_fingerprint(df1, df2, weight1, weight2):
    columns_to_process = [col for col in df1.columns if col.isdigit() and 500 <= int(col) <= 1500]

    df_avg = df2.copy()

    df_avg[columns_to_process] = (df1[columns_to_process] * weight1 + df2[columns_to_process] * weight2) / (weight1+weight2)

    df_avg.to_csv("./test_ensemble_fingerprint.csv", index=False)

df2 = pd.read_csv("./output/model/morgan_classical/fold_0/classical.csv")
df1 = pd.read_csv("./output/model/morgan_hybrid_fingerprint/fold_0/morgan_hybrid_fingerprint.csv")

ensemble_fingerprint(df1, df2, 6, 4)

