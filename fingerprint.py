import pandas as pd

df = pd.read_csv("/workspaces/project/data/research_data/test_full.csv")

columns_to_keep = [col for col in df.columns if col.isdigit() and 500 <= int(col) <= 1500 and int(col) % 2 == 0]

df_selected = df[["smiles"] + columns_to_keep] if "smiles" in df.columns else df[columns_to_keep]

df_selected.to_csv("/workspaces/project/data/research_fingerprint_data/test_full.csv", index=False)


