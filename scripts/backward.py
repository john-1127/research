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

    print(count_k)

if __name__ == '__main__':
    main()
