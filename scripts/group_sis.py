import pandas as pd
import sys
import os

def extract_sis_column(txt_path, model_name, group_id):
    df = pd.read_csv(txt_path, sep=None, engine='python')
    sis_col = df.columns[-1]
    df = df[[sis_col]].copy()
    df.columns = ['SIS']
    df['Model'] = model_name
    df['Group'] = group_id
    return df[['Group', 'Model', 'SIS']]

def merge_sis_files(ffnn_path, hqnn_path, ensemble_path, group_id, output_path):
    df_ffnn = extract_sis_column(ffnn_path, 'FFNN', group_id)
    df_hqnn = extract_sis_column(hqnn_path, 'HQNN', group_id)
    df_ensemble = extract_sis_column(ensemble_path, 'Ensemble', group_id)

    df_all = pd.concat([df_ffnn, df_hqnn, df_ensemble], ignore_index=True)
    df_all.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))
    print(f"Saved combined SIS data to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python group_sis.py ffnn.txt hqnn.txt ensemble.txt group_id")
    else:
        ffnn_path = f"./output/sis/{sys.argv[1]}.txt"
        hqnn_path = f"./output/sis/{sys.argv[2]}.txt"
        ensemble_path = f"./output/sis/{sys.argv[3]}.txt"
        group_id = f"Group{sys.argv[4]}"
        output_path = f"./output/figures/distribution_group4_7.csv"
        merge_sis_files(ffnn_path, hqnn_path, ensemble_path, group_id, output_path)
