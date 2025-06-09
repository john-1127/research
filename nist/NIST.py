import pandas as pd
import requests
import time
import csv

df = pd.read_csv("./data/research_data/train_smiles.csv")

with open("smiles_with_cas.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["smiles", "cas"])

    for i, smiles in enumerate(df["smiles"]):
        try:
            url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/cas"
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and response.text.strip():
                cas = response.text.strip().split('\n')[0]
            else:
                cas = "NA"
        except Exception:
            cas = "NA"

        writer.writerow([smiles, cas])
        print(f'{i}: {cas}')
        time.sleep(1)

