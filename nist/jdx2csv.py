import pandas as pd
import os
import numpy as np


df = pd.read_csv('./nist/test_cas.csv')
jcamp_dir = './nist/jcamp'


def read_jdx(fn):
    FIRSTX = DELTAX = NPOINTS = None
    Y = []
    with open(fn, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('##FIRSTX='):
                FIRSTX = float(line.split('=',1)[1])
            elif line.startswith('##DELTAX='):
                DELTAX = float(line.split('=',1)[1])
            elif line.startswith('##NPOINTS='):
                NPOINTS = int(line.split('=',1)[1])
            elif line.startswith('##YUNITS='):
                YUNITS = line.split('=',1)[1].strip().upper()
            elif line.startswith('##XYDATA'):
                # 一旦到 XYDATA，就跳出去開始讀下面的數據
                break

        if YUNITS != 'ABSORBANCE':
            print(f'{YUNITS}: Not an absorbance!')
            return None, None

        elif DELTAX != 4.0:
            print('Bad delta')
            return None, None

        # 開始讀 data 區塊，每行第一個值是當前 X，其後才是多個 Y
        for line in f:
            if line.startswith('##'):  # 碰到下一個段落就結束
                break
            parts = line.strip().split()
            if not parts:
                continue
            ys = [float(v) for v in parts[1:]]  # parts[0] 是 X，本行的 Y 值從 parts[1:] 開始
            Y.extend(ys)
            if NPOINTS and len(Y) >= NPOINTS:
                Y = Y[:NPOINTS]
                break

    X = [FIRSTX + i*DELTAX for i in range(len(Y))]
    return X, Y


grid = np.arange(400, 4000+1, 2)
out_rows = [] 
smiles_list = []

for fname in os.listdir(jcamp_dir):
    id_str = os.path.splitext(fname)[0]
    try:
        idx = int(id_str)
         
    except ValueError:
        continue

    path = os.path.join(jcamp_dir, fname)
    X, Y = read_jdx(path)
    if X is None:
        continue

    spec = dict(zip(X, Y))

    try:
        smiles = df.loc[idx, 'smiles']
    except KeyError:
        print('SMILES not found')
        smiles = None

    smiles_list.append(smiles)
    row = {'smiles': smiles}
    for w in grid:
        row[w] = spec.get(w, np.nan)

    for w in grid:
        if row[w] == 0:
            row[w] = 1e-10

    finger_ws = [w for w in grid if 500 <= w <= 1500]
    s_finger = sum(row[w] for w in finger_ws if not np.isnan(row[w]))
    if s_finger > 0:
        for w in finger_ws:
            if not np.isnan(row[w]):
                row[w] = row[w] / s_finger

    # all_ws = [w for w in grid]
    # # s_all = sum(row[w] for w in all_ws if not np.isnan(row[w]))
    # s_max = max(row[w] for w in all_ws if not np.isnan(row[w]))
    # if s_max > 0:
    #     for w in all_ws:
    #         if not np.isnan(row[w]):
    #             row[w] = row[w] / s_max
    #
    out_rows.append(row)

result = pd.DataFrame(out_rows)
result.to_csv('./nist/test_spectra.csv', index=False, na_rep='nan')
pd.DataFrame({'smiles': smiles_list}).to_csv('./nist/predict_smiles.csv', index=False, encoding='utf-8')
