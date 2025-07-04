import os
import pandas as pd

def extract_deltax_from_jdx(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            if line.startswith('##DELTAX='):
                deltax_value = line.split('=')[1].strip()
                return float(deltax_value)
    return None

def check_deltax_in_folder(folder_path):
    result = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jdx'):
            filepath = os.path.join(folder_path, filename)
            deltax = extract_deltax_from_jdx(filepath)
            result.append((filename, deltax))
    return result

folder = './nist/jcamp'

print(len(check_deltax_in_folder(folder)))
df = pd.read_csv('./nist/test_spectra.csv')
test = df.loc[0, '400']
print(test)
