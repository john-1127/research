import pandas as pd
import subprocess
import sys
import time
import shutil
import os
import numpy as np
from scipy.interpolate import CubicSpline

log_path = "./fine-tune/log.txt"
# sys.stdout = open(log_path, "a", encoding="utf-8")


def read_meta(fn):
    meta = {
        "state": None,
        "xunits": None,
        "yunits": None,
        "deltax": None,
        "xfactor": None,
        "yfactor": None,
        "xydata": None,
        "x": [],
        "y": [],
    }
    
    with open(fn, "r", encoding="utf-8", errors="ignore") as f:
        parsing_data = False
        next_x = None

        for line in f:
            line = line.strip()

            # Meta Data
            if line.startswith('##STATE='):
                meta['state'] = line.split('=', 1)[1].lower()

            elif line.startswith('##XUNITS='):
                meta['xunits'] = line.split('=', 1)[1]

            elif line.startswith('##YUNITS='):
                meta['yunits']= line.split('=', 1)[1].lower()

            elif line.startswith('##DELTAX='):
                meta['deltax'] = float(line.split('=', 1)[1])

            elif line.startswith('##XFACTOR='):
                meta['xfactor'] = float(line.split('=', 1)[1])

            elif line.startswith('##YFACTOR='):
                meta['yfactor'] = float(line.split('=', 1)[1])

            elif line.startswith('##XYDATA='):
                meta['xydata'] = line.split('=', 1)[1]
                parsing_data = True
                continue 

            if parsing_data and line and not line.startswith('##'):
                parts = line.split()
                if not parts:
                    continue
                try:
                    x_start = float(parts[0]) * meta['xfactor']
                    ys = [float(y) * meta['yfactor'] for y in parts[1:]]

                    for i, y in enumerate(ys):
                        # 使用 deltax 或直接給的 x_start
                        x = round(x_start + i * meta['deltax'], 3) if i > 0 else round(x_start, 3)
                        meta['x'].append(x)
                        meta['y'].append(y)
                except Exception as e:
                    print(f"{fn}")
                    print(f"Error parsing line: {line} -> {e}")

    return meta

def interpolate_even_x(x_list, y_list):
    x_start = min(x_list)
    x_end = max(x_list)

    new_start = int(np.ceil(x_start))
    if new_start % 2 != 0:
        new_start += 1

    new_end = int(np.floor(x_end))
    if new_end % 2 != 0:
        new_end -= 1

    new_x = np.arange(new_start, new_end + 1, 2)

    cs = CubicSpline(x_list, y_list, bc_type='natural')
    new_y = cs(new_x)

    return new_x, new_y

if __name__ == '__main__':

    
    df = pd.read_csv("./fine-tune/data_result.csv")
    dir_path = "./fine-tune/data/gas/absorbance"            

    all_rows = []
    for file in os.listdir(dir_path):
        idx = -1
        if file.endswith(".jdx"):
            filename = os.path.splitext(file)[0]
            try:
                idx = int(filename)

            except ValueError:
                print(f"Error, can't convert filename：{filename}")

        if idx != -1:
            smiles = df.iloc[idx]['smiles']
            meta = read_meta(os.path.join(dir_path, file))

            # Check Data Format
            if meta['state'] != 'gas' or meta['xunits'] != '1/CM' or meta['yunits'] != 'absorbance' \
                or meta['xydata'] != '(X++(Y..Y))' or meta['xfactor'] != 1.0 or meta['yfactor'] == None:
                print(f"Error, wrong data format: {filename}")
                break

            if not np.all(np.diff(meta['x']) > 0):
                print(f"{idx}: is not strictly increasing")

            try:
                x, y = interpolate_even_x(meta['x'], meta['y'])

            except Exception as e:
                print(f"Error: {idx}")
                print(f"{meta['x'][:10]}")
                break

            full_x = np.arange(400, 4001, 2)
            full_y = [np.nan] * len(full_x)

            x_to_y = dict(zip(x, y))
            for i , x_val in enumerate(full_x):
                if x_val in x_to_y:
                    val = x_to_y[x_val]
                    if val <= 0:
                        val = 1e-10
                    full_y[i] = val

            row = [smiles] + full_y
            all_rows.append(row)

    columns = ['smiles'] + full_x.tolist()
    df_1 = pd.DataFrame(all_rows, columns=columns) 

    df_1.to_csv("./fine-tune/test.csv", index=False, na_rep="NaN")
