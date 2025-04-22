import pandas as pd
import sys
import os

def ensemble(df1, df2, output_path, weight1=5, weight2=5):

    columns_to_process = [col for col in df1.columns if col.isdigit() and 400 <= int(col) <= 4000]

    df_avg = df1.copy()
    df_avg[columns_to_process] = (df1[columns_to_process] * weight1+ df2[columns_to_process] * weight2) / (weight1 + weight2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_avg.to_csv(output_path, index=False)


def ensemble_fingerprint(df1, df2, weight1, weight2):
    columns_to_process = [col for col in df1.columns if col.isdigit() and 500 <= int(col) <= 1500]

    df_avg = df2.copy()

    df_avg[columns_to_process] = (df1[columns_to_process] * weight1 + df2[columns_to_process] * weight2) / (weight1+weight2)

    df_avg.to_csv("./test_ensemble_fingerprint.csv", index=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python ensemble.py classical.csv hqnn.csv")
    else:
        df1_path = f"./output/model/{sys.argv[1]}/fold_0/{sys.argv[1]}.csv"
        df2_path = f"./output/model/{sys.argv[2]}/fold_0/{sys.argv[2]}.csv"
        output_path = f"./output/model/ensemble_{sys.argv[2]}/fold_0/ensemble_{sys.argv[2]}.csv"

        df1 = pd.read_csv(df1_path)
        df2 = pd.read_csv(df2_path)

        ensemble(df1, df2, output_path=output_path, weight1=5, weight2=5)
