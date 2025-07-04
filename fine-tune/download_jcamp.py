import pandas as pd
import subprocess
import sys
import time

log_path = "./fine-tune/log.txt"
sys.stdout = open(log_path, "a", encoding="utf-8")

df = pd.read_csv("./fine-tune/data_result.csv")

for idx, row in df.iterrows():

    print(f"Index: {idx}, SMILES: {row['smiles']}, URL: {row['url']}")

    smiles = row['smiles']
    url = row['url']

    if all(x not in smiles for x in ['.', '+', '-']): 
        try:
            result = subprocess.run(["wget", "-O", f"./fine-tune/data/{idx}.jdx", url], check=True)
            time.sleep(0.5)

        except subprocess.CalledProcessError:
            print(f"Error: Failed to download, (Index {idx}) - skipped")
            continue
