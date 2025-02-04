import pandas as pd


# full 是答案
path = [
    "./chempropIRZenodo/trained_ir_model/computed_model/test_preds.csv",
    "./chempropIRZenodo/trained_ir_model/computed_model/test_full.csv",
    "./chempropIRZenodo/trained_ir_model/computed_model/test_smiles.csv",
    "./chempropIRZenodo/trained_ir_model/computed_model/computed_spectra.csv",
    "./output/fingerprint_preds.csv",
]

df = pd.read_csv(path[3])

print(df.head())
print(df.shape)
# print(df[df['smiles'] == 'CCCC(C)Oc1nc(N)c2nc(OC)n(CCCCC3CCCNC3)c2n1'])
# print(df.iloc[0])

