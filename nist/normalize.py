import pandas as pd
import numpy as np

def normalize_spectra(csv_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    spectrum_cols = [col for col in df.columns if col != 'smiles']

    # 每一列 normalize
    for idx, row in df.iterrows():
        spectrum = row[spectrum_cols].astype(float)
        total = spectrum.max()

        if total > 0:
            normalized = spectrum / total
            df.loc[idx, spectrum_cols] = normalized


    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Normalized data saved to {output_path}")

    return df

normalize_spectra("./nist/predict.csv", "./nist/predict_normalize.csv")
