#!/usr/bin/python3

import math
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols 

"""
Script for calculating SIS between two spectra csv files, usually the model predictions and the test set targets.
Input as python SIS_similarity.py path_to_file1 path_to_file2
"""

log_path = "./log_3.txt"
sys.stdout = open(log_path, "a", encoding="utf-8")

groups = {
    "alcohols_OH": "[#6][OX2H]",
    "1_2_amine_NH": "[NX3;H2,H1]",
    "alkyne_CH": "[CX2]#[CX2;!H0]",
    "alkene_CH": "[CX3]=[CX3;!H0]",
    "aromatic_CH": "[cH]",
    "carboxylic_OH": "[CX3](=O)[OX2H1]",
    "alkane_CH": "[CX4;!H0]",
    "aldehyde_CH": "[CX3H1](=O)",
    "nitrile_CN": "[NX1]#[CX2]",
    "alkyne_CC": "[CX2]#[CX2]",
    "C=O": "[CX3]=[OX1]",
    "alkene_CC": "[CX3]=[CX3]",
    "1_amine_NH": "[NX3;H2;!$(NC=O)]",
    "aromatic_CC": "[c]:[c]",
    "nitro_asym_NO": "[!#8][NX3+](=O)[O-]",
    "nitro_sym_NO": "[!#8][NX3](=O)=O",
    "C-O": "C[OX2]",
    "Haloalkane_CH": "[CX4;!H0][F,Cl,Br,I]",
    "alphatic_amine_CN": "C[NX3;!$(NC=O)]",
    "alkyl_Cl_Br_CX": "[#6][Cl,Br]",
    "alkyl_F": "[#6][F]",
    "Si_O": "[Si]-[O]",
    "Si=O": "[Si](=O)",
    "C_Si": "[#6][Si]",
    "P_O": "[P]-[O]",
    "P=O": "P(=O)",
    "C_P": "[#6]P", 
}

def spectral_information_similarity(spectrum1,spectrum2,conv_matrix,frequencies=list(range(400,4002,2)),threshold=1e-10,std_dev=10):
    length = len(spectrum1)
    nan_mask=np.isnan(spectrum1)+np.isnan(spectrum2)
    # print(length,conv_matrix.shape,spectrum1.shape,spectrum2.shape)
    assert length == len(spectrum2), "compared spectra are of different lengths"
    assert length == len(frequencies), "compared spectra are a different length than the frequencies list, which can be specified"
    spectrum1[spectrum1<threshold]=threshold
    spectrum2[spectrum2<threshold]=threshold
    spectrum1[nan_mask]=0
    spectrum2[nan_mask]=0
    # print(spectrum1.shape,spectrum2.shape)
    spectrum1=np.expand_dims(spectrum1,axis=0)
    spectrum2=np.expand_dims(spectrum2,axis=0)
    # print(spectrum1.shape,spectrum2.shape)
    conv1=np.matmul(spectrum1,conv_matrix)
    # print(conv1[0,1000])
    conv2=np.matmul(spectrum2,conv_matrix)
    conv1[0,nan_mask]=np.nan
    conv2[0,nan_mask]=np.nan
    # print(conv1.shape,conv2.shape)
    sum1=np.nansum(conv1)
    sum2=np.nansum(conv2)
    norm1=conv1/sum1
    norm2=conv2/sum2
    distance=norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1)
    sim=1/(1+np.nansum(distance))
    return sim

def import_smiles(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        smiles=[]
        for row in r:
            smiles.append(row[0])
        return smiles

def import_data(file):
    with open(file,'r') as rf:
        r=csv.reader(rf)
        next(r)
        data=[]
        for row in r:
            data.append(row)
        return data

def make_conv_matrix(frequencies=list(range(400,4002,2)),std_dev=10):
    length=len(frequencies)
    gaussian=[(1/(2*math.pi*std_dev**2)**0.5)*math.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2)) for i in range(length)]
    conv_matrix=np.empty([length,length])
    for i in range(length):
        for j in range(length):
            conv_matrix[i,j]=gaussian[abs(i-j)]
    return conv_matrix

def functional_count(smiles, smarts):
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(patt)
    return len(matches)

def count_all_functional_groups(smiles):
    result = []
    for group_name, smarts in groups.items():
        count = functional_count(smiles, smarts)
        result.append(count)
    return result


def preprocessing(group_counts):
    ignored_groups = {
        "alkyne_CH",
        "alkene_CH",
        "aromatic_CH",
        "alkane_CH",
        "aldehyde_CH",
    }

    group_order = list(groups.keys())
    result = []
    i = 0
    while i < len(group_order):
        group_name = group_order[i]
        if group_name in ignored_groups:
            i += 1
            continue
        # Merge nitro
        elif group_name == "nitro_asym_NO":
            count = group_counts[i] + group_counts[i + 1]
            result.append(count)
            i += 2

        else:
            result.append(group_counts[i])
            i += 1

    # Max Scaling
    max_val = max(result)
    if max_val == 0:
        return result

    return [v / max_val for v in result]

def preprocessing_merge(group_counts):
    ignored_groups = {
        "alkyne_CH",
        "alkene_CH",
        "aromatic_CH",
        "alkane_CH",
        "aldehyde_CH",
    }

    merged = {
        "amine_alcohol_mix": 0,
        "amine_aromatic_NO": 0,
        "CO_amines": 0,
        "amine_alkyl_mix": 0
    }

    group_order = list(groups.keys())
    group_idx = {name: i for i, name in enumerate(group_order)}
    result = []

    for name, idx in group_idx.items():
        count = group_counts[idx]

        # if name in ignored_groups:
        #     continue

        # if name in {"1_2_amine_NH", "alcohols_OH"}:
        #     merged["amine_alcohol_mix"] += count
        # if name in {"1_amine_NH", "aromatic_CC", "nitro_asym_NO", "nitro_sym_NO"}:
        #     merged["amine_aromatic_NO"] += count
        # if name in {"C-O", "alphatic_amine_CN"}:
        #     merged["CO_amines"] += count
        # if name in {"1_2_amine_NH", "alkyl_Cl_Br_CX"}:
        #     merged["amine_alkyl_mix"] += count
        #
        # if name not in {
        #     "1_2_amine_NH", "alcohols_OH", "1_amine_NH", "aromatic_CC",
        #     "nitro_asym_NO", "nitro_sym_NO", "C-O", "alphatic_amine_CN", "alkyl_Cl_Br_CX"
        # }:
        result.append(count)

    # result.append(merged["amine_alcohol_mix"])
    # result.append(merged["amine_aromatic_NO"])
    # result.append(merged["CO_amines"])
    # result.append(merged["amine_alkyl_mix"])

    # Max Scaling
    max_val = max(result)
    if max_val == 0:
        return result

    return [v / max_val for v in result] 

def cosine_similarity(smiles1, smiles2):
    vec1 = count_all_functional_groups(smiles1)
    vec2 = count_all_functional_groups(smiles2)
    process_vec1 = preprocessing_merge(vec1)
    process_vec2 = preprocessing_merge(vec2)
    v1 = np.array(process_vec1)
    v2 = np.array(process_vec2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return [0.0, process_vec1, process_vec2]

    return [np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), process_vec1, process_vec2]

def matching_similarity(smiles1, smiles2):
    vec1 = count_all_functional_groups(smiles1)
    vec2 = count_all_functional_groups(smiles2)
    process_vec1 = preprocessing_merge(vec1)
    process_vec2 = preprocessing_merge(vec2)
    v1 = np.array(process_vec1)
    v2 = np.array(process_vec2)

    match = 0
    for i,j in zip(v1, v2):
        if i==j:
            match += 1

    return [match, process_vec1, process_vec2]

def L1_distance(smiles1, smiles2):
    vec1 = count_all_functional_groups(smiles1)
    vec2 = count_all_functional_groups(smiles2)
    process_vec1 = preprocessing_merge(vec1)
    process_vec2 = preprocessing_merge(vec2)
    v1 = np.array(process_vec1)
    v2 = np.array(process_vec2)

    return [np.sum(np.abs(v1 - v2)), process_vec1, process_vec2]



def main():
    smiles_pred  = import_smiles(sys.argv[1])
    spectra_pred = import_data(sys.argv[1])
    spectra_ref  = import_data(sys.argv[2])

    conv_matrix = make_conv_matrix()
    count_k = 0
    ex_count = 0
    # bad_hit = 0
    k = 1

    sis_scores = {}
    with open("./output/sis/qnn_pretrained.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sis_scores[parts[0]] = float(parts[1])

    for pred_row, pred_smiles in zip(spectra_pred, smiles_pred):
        pred_spec = np.array(pred_row[1:], dtype=float)

        max_sim = 0
        ans = ''
        for ref_row in spectra_ref:
            ref_smiles = ref_row[0]
            ref_spec = np.array(ref_row[1:], dtype=float)

            sim = spectral_information_similarity(pred_spec, ref_spec, conv_matrix)
            if sim > max_sim:
                max_sim = sim
                ans = ref_smiles

        # Hit
        if pred_smiles == ans:
            count_k += 1

        # Ex_Hit
        else:
            score, vec1, vec2 = L1_distance(pred_smiles, ans)
            if score <= 1:
                print(f"{pred_smiles}, {ans}, {score}, {sis_scores[pred_smiles]}")
                ex_count += 1

    print(f"Top-{k} 命中數: {count_k}")
    print(f"額外命中數: {ex_count}")
    Acc = round(100* (count_k+ex_count) / 748, 1)
    print(f"Acc: {Acc}%")


if __name__ == '__main__':
    main()
