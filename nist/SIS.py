#!/usr/bin/python3

import math
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

"""
Script for calculating SIS between two spectra csv files, usually the model predictions and the test set targets.
Input as python SIS_similarity.py path_to_file1 path_to_file2
"""

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

def rmse_mae(spectrum1,spectrum2):
    length=len(spectrum1)
    rmse=0.0
    mae=0.0
    for x in range(length):
        rmse+=(spectrum1[x]-spectrum2[x])**2
        mae+=abs(spectrum1[x]-spectrum2[x])
    rmse/=length
    rmse==rmse**0.5
    mae/=length
    return rmse,mae

def make_conv_matrix(frequencies=list(range(400,4002,2)),std_dev=10):
    length=len(frequencies)
    gaussian=[(1/(2*math.pi*std_dev**2)**0.5)*math.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2)) for i in range(length)]
    conv_matrix=np.empty([length,length])
    for i in range(length):
        for j in range(length):
            conv_matrix[i,j]=gaussian[abs(i-j)]
    return conv_matrix

def heat_map(top_k_smiles, spectra_ref, conv_matrix):

    rows = []
    sims = []
    for ref_row in spectra_ref:
        ref_smiles = ref_row[0]
        ref_spec = np.array(ref_row[1:], dtype=float)
        if ref_smiles in top_k_smiles:
            rows.append(ref_row)
    
    for row in rows:
        row_smiles = row[0] 
        row_spec = np.array(row[1:], dtype=float)

        for i in rows:
            i_smiles = i[0]
            i_spec = np.array(i[1:], dtype=float)
            sim = spectral_information_similarity(row_spec, i_spec, conv_matrix)
            sims.append(sim)

def heat_map(top_k_smiles, spectra_ref, conv_matrix,
             frequencies=list(range(400,4002,2)),
             threshold=1e-10, std_dev=10):
    # 1. 先挑出要比對的 spectra rows
    filtered = []
    for smiles, *spec in spectra_ref:
        if smiles in top_k_smiles:
            filtered.append((smiles, np.array(spec, dtype=float)))
    k = len(filtered)

    # 2. 建立 k×k 的相似度矩陣
    sim_matrix = np.zeros((k, k), dtype=float)
    for i, (smi_i, spec_i) in enumerate(filtered):
        for j, (smi_j, spec_j) in enumerate(filtered):
            sim = spectral_information_similarity(
                spec_i, spec_j, conv_matrix,
                frequencies=frequencies,
                threshold=threshold,
                std_dev=std_dev
            )
            sim_matrix[i, j] = sim

    # 3. 畫 heatmap
    # labels = [smi for smi, _ in filtered]
    indices = list(range(1,k+1))
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(sim_matrix, interpolation='nearest', aspect='equal')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(indices, rotation=90, fontsize=8)
    ax.set_yticklabels(indices, fontsize=8)
    ax.set_title(f"Top-{k} Spectral Similarity Heatmap")
    fig.colorbar(cax, ax=ax, label='Similarity')
    fig.savefig('heatmap.png',dpi=300,bbox_inches='tight')

    return sim_matrix

def main():
    smiles_pred   = import_smiles(sys.argv[1])
    spectra_pred  = import_data(sys.argv[1])
    spectra_ref   = import_data(sys.argv[2])

    conv_matrix = make_conv_matrix()

    count_k = 0
    k = 5

    for pred_row, pred_smiles in zip(spectra_pred, smiles_pred):
        pred_spec = np.array(pred_row[1:], dtype=float)

        sims = []
        for ref_row in spectra_ref:
            ref_smiles = ref_row[0]
            ref_spec   = np.array(ref_row[1:], dtype=float)

            sim = spectral_information_similarity(pred_spec,
                                                   ref_spec,
                                                   conv_matrix)
            sims.append((ref_smiles, sim))

        top_k = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
        for smile in top_k:
            print(smile)

        top_k_smiles = [sm for sm, _ in top_k]
        # heat_map(top_k_smiles, spectra_ref, conv_matrix)

        if pred_smiles in top_k_smiles:
            count_k += 1

        print("")
    print(count_k)

if __name__ == '__main__':
    main()
