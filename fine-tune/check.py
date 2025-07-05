import pandas as pd

df = pd.read_csv("./data/research_data/experiment.csv")

spectrum_cols = [str(i) for i in range(400, 4001, 2)]
spectrum_data = df[spectrum_cols]

nan_ratios = spectrum_data.isna().sum(axis=1) / len(spectrum_cols)

for i, ratio in enumerate(nan_ratios):
    missing_count = int(ratio * len(spectrum_cols))
    
    if ratio > 10:
        print(f"Row {i}: {missing_count}/1801 ({ratio:.2%})")

