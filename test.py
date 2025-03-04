import pandas as pd


# full 是答案
path = [
    "./chempropIRZenodo/trained_ir_model/computed_model/test_preds.csv",
    "./chempropIRZenodo/trained_ir_model/computed_model/test_full.csv",
    "./chempropIRZenodo/trained_ir_model/computed_model/test_smiles.csv",
    "./chempropIRZenodo/trained_ir_model/computed_model/computed_spectra.csv",
    "./output/model_classical/classical.csv",
    "./data/computed_data/train_full.csv"
]

df = pd.read_csv(path[5])

# print(df.head())
print(df.iloc[1527].getitem(0))
# print(df.shape)
# print(df[df['smiles'] == 'CCCC(C)Oc1nc(N)c2nc(OC)n(CCCCC3CCCNC3)c2n1'])
# print(df.iloc[0])

