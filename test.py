import pandas as pd


path = [
    "./data/research_data/test_full.csv",
    "./data/research_fingerprint_data/test_full.csv",
    "./output/sis/sis_summary.csv"
]

df = pd.read_csv(path[2])
# print(df.head())
print(df.to_string())
# print(df.shape)
# print(df[df['smiles'] == 'CCCC(C)Oc1nc(N)c2nc(OC)n(CCCCC3CCCNC3)c2n1'])
# print(df.iloc[0])

