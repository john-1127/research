import pandas as pd


path = [
    "./data/research_data/test_full.csv",
    "./data/research_fingerprint_data/test_full.csv",
    "./output/model/classical_2100_layer3/fold_0/classical_2100_layer3.csv",
    "./output/sis/sis_summary.csv"
]

df = pd.read_csv(path[3])
# print(df.head())
print(df.to_string())

df1 = pd.read_csv(path[0])
print(df1.head())
# print(df.shape)
# print(df[df['smiles'] == 'CCCC(C)Oc1nc(N)c2nc(OC)n(CCCCC3CCCNC3)c2n1'])
# print(df.iloc[0])

