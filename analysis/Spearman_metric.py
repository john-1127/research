import pandas as pd

filename = "./analysis/metric_summary.csv"

df = pd.read_csv(filename)

columns = ['Spearman Average', 'Spearman Q1', 'Spearman Median', 'Spearman Q3']
selected = df[columns]

rounded = selected.round(4)

for i, row in rounded.iterrows():
    if i % 3 == 2:
        print(row.tolist())
