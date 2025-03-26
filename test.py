import pandas as pd


path = [
    "./data/research_data/test_full.csv",
    "./data/research_fingerprint_data/test_full.csv"
]

df = pd.read_csv(path[1])
df1 = pd.read_csv(path[0])

print(df.head())
print(df1.head())
print(df.shape)
print(df1.shape)
# print(df[df['smiles'] == 'CCCC(C)Oc1nc(N)c2nc(OC)n(CCCCC3CCCNC3)c2n1'])
# print(df.iloc[0])

