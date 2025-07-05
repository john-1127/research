import pandas as pd

def extract_sis(file_path, model_name):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            sis = float(line.split()[-1]) 
            data.append({"Model": model_name, "SIS": sis})
        except ValueError:
            print(f"Skipped: {line}")
    return data

# data_direct = extract_sis("./output/sis/ffnn_pretrained.txt", "FFNN")
# data_pretrained = extract_sis("./output/sis/qnn_pretrained.txt", "HQNN")
# data_ensemble = extract_sis("./output/sis/ensemble_qnn_pretrained.txt", "Ensemble")
#
#
# df = pd.DataFrame(data_direct + data_pretrained + data_ensemble)
# df.to_csv("./output/sis/sis_summary_pretrained_ffnn_vs_qnn_vs_ensemble.csv", index=False)


df = pd.read_csv("./output/sis/sis_summary_pretrained.csv")

df.insert(0, "Group", 1)

df.to_csv("./output/sis/direct_pretrained.csv", index=False)

