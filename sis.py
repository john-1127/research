import pandas as pd

df = pd.read_csv("./output/sis/similarity.txt", sep=r"\s+",header=None) 


avg_sis = df[1].mean()

print(f"SIS 平均值: {avg_sis:.6f}")

