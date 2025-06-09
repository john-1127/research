def get_worst_10_percent(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        smiles, score = parts[0], float(parts[1])
        data.append((smiles, score))

    data.sort(key=lambda x: x[1])
    num_to_select = max(1, int(len(data)))
    return data[:num_to_select]


def remove_common_smiles(data1, data2):
    smiles1 = set([x[0] for x in data1])
    smiles2 = set([x[0] for x in data2])
    common = smiles1 & smiles2

    filtered1 = [x for x in data1 if x[0] not in common]
    filtered2 = [x for x in data2 if x[0] not in common]

    return filtered1, filtered2


def save_to_file(data1, data2, output_path):
    with open(output_path, 'w') as f:
        f.write("### HQNN 最差 1%（去除重複後） ###\n")
        for smiles, score in data1:
            f.write(f"{smiles} {score}\n")

        f.write("\n### FFNN 最差 1%（去除重複後） ###\n")
        for smiles, score in data2:
            f.write(f"{smiles} {score}\n")

def find_ffnn_bad_qnn_good(data_qnn, data_ffnn, threshold, diff, output_path):
    qnn_dict = {smiles: score for smiles, score in data_qnn}
    ffnn_dict = {smiles: score for smiles, score in data_ffnn}

    result = []

    for smiles, ffnn_score in ffnn_dict.items():
        if ffnn_score < threshold:
            qnn_score = qnn_dict.get(smiles)
            if qnn_score is not None and (qnn_score - ffnn_score) >= diff:
                result.append((smiles, qnn_score, ffnn_score))

    with open(output_path, 'w') as f:
        f.write("smiles HQNN_SIS FFNN_SIS\n")
        for smiles, qnn_s, ffnn_s in result:
            f.write(f"{smiles} {qnn_s:.4f} {ffnn_s:.4f}\n")


def find_hqnn_bad_ffnn_good(data_qnn, data_ffnn, threshold, diff, output_path):
    qnn_dict = {smiles: score for smiles, score in data_qnn}
    ffnn_dict = {smiles: score for smiles, score in data_ffnn}

    result = []

    for smiles, qnn_score in qnn_dict.items():
        if qnn_score < threshold:
            ffnn_score = ffnn_dict.get(smiles)
            if ffnn_score is not None and (ffnn_score - qnn_score) >= diff:
                result.append((smiles, ffnn_score, qnn_score))

    with open(output_path, 'w') as f:
        f.write("smiles FFNN_SIS HQNN_SIS\n")
        for smiles, qnn_s, ffnn_s in result:
            f.write(f"{smiles} {qnn_s:.4f} {ffnn_s:.4f}\n")


worst_qnn = get_worst_10_percent('./output/sis/qh2_2100_layer3.txt')
worst_ffnn = get_worst_10_percent('./output/sis/classical_2100_layer3.txt')

unique1, unique2 = remove_common_smiles(worst_qnn, worst_ffnn)

threshold = 0.7
diff = 0.15
find_ffnn_bad_qnn_good(
    data_qnn=worst_qnn,
    data_ffnn=worst_ffnn,
    threshold=threshold,
    diff=diff,
    output_path='./analysis/ffnn_bad_qnn_good.txt'
)
find_hqnn_bad_ffnn_good(
    data_qnn=worst_qnn,
    data_ffnn=worst_ffnn,
    threshold=threshold,
    diff=diff,
    output_path='./analysis/hqnn_bad_ffnn_good.txt'
)

# save_to_file(unique1, unique2, './analysis/worst_smiles.txt')


