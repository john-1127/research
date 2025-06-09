import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr


def calculate_tmse(reference_csv, predict_csv):
    ref_df = pd.read_csv(reference_csv)
    pred_df = pd.read_csv(predict_csv)

    assert ref_df.shape == pred_df.shape, "Reference and prediction CSV must have the same shape."
    assert (ref_df['smiles'] == pred_df['smiles']).all(), "SMILES entries do not match between files."

    spectrum_cols = ref_df.columns[1:]
    tmse_list = []

    for i in range(len(ref_df)):
        smiles = ref_df.iloc[i]['smiles']
        y_true = ref_df.iloc[i][spectrum_cols].astype(float).values
        y_pred = pred_df.iloc[i][spectrum_cols].astype(float).values

        error = ((y_pred - y_true) ** 2) / (y_true)
        tmse = np.mean(error)

        tmse_list.append({
            'smiles': smiles,
            'TMSE': tmse
        })

    return pd.DataFrame(tmse_list)


def calculate_spearman(reference_csv, predict_csv):
    ref_df = pd.read_csv(reference_csv)
    pred_df = pd.read_csv(predict_csv)

    assert ref_df.shape == pred_df.shape, "CSV shape mismatch"
    assert (ref_df['smiles'] == pred_df['smiles']).all(), "SMILES mismatch"

    spectrum_cols = ref_df.columns[1:]
    spearman_list = []

    for i in range(len(ref_df)):
        smiles = ref_df.iloc[i]['smiles']
        y_true = ref_df.iloc[i][spectrum_cols].astype(float).values
        y_pred = pred_df.iloc[i][spectrum_cols].astype(float).values

        corr, _ = spearmanr(y_true, y_pred)
        spearman_list.append({
            'smiles': smiles,
            'Spearman': corr
        })

    return pd.DataFrame(spearman_list)


def calculate_pearson(reference_csv, predict_csv):
    ref_df = pd.read_csv(reference_csv)
    pred_df = pd.read_csv(predict_csv)

    assert ref_df.shape == pred_df.shape, "CSV shape mismatch"
    assert (ref_df['smiles'] == pred_df['smiles']).all(), "SMILES mismatch"

    spectrum_cols = ref_df.columns[1:]
    pearson_list = []

    for i in range(len(ref_df)):
        smiles = ref_df.iloc[i]['smiles']
        y_true = ref_df.iloc[i][spectrum_cols].astype(float).values
        y_pred = pred_df.iloc[i][spectrum_cols].astype(float).values

        corr, _ = pearsonr(y_true, y_pred)
        pearson_list.append({
            'smiles': smiles,
            'Pearson': corr
        })

    return pd.DataFrame(pearson_list)

# if __name__ == '__main__':
#     reference_csv = './data/research_data/test_full.csv'
#
#     predicts = [
#         'classical_300_layer3', 'qh8_300_layer3', 'ensemble_qh8_300_layer3',
#         'classical_1200_layer3', 'qh8_1200_layer3', 'ensemble_qh8_1200_layer3',
#         'classical_2100_layer3', 'qh2_2100_layer3', 'ensemble_qh2_2100_layer3',
#         'classical_300_layer2', 'qh2_300_layer2', 'ensemble_qh2_300_layer2',
#         'classical_1200_layer2', 'qh8_1200_layer2', 'ensemble_qh8_1200_layer2',
#         'classical_2100_layer2', 'qh2_2100_layer2', 'ensemble_qh2_2100_layer2',
#         'classical_2048_layer1', 'qh2_2048_layer1', 'ensemble_qh2_2048_layer1'
#     ]
#
#     model_cycle = ['FFNN', 'HQNN', 'Ensemble']
#     results = []
#
#     for idx, predict in enumerate(predicts):
#         group_id = f'{idx // 3 + 1}'
#         model_type = model_cycle[idx % 3]
#         predict_csv = f'./output/model/{predict}/fold_0/{predict}.csv'
#
#         spearman_df = calculate_spearman(reference_csv, predict_csv)
#
#         for _, row in spearman_df.iterrows():
#             results.append({
#                 'Group': group_id,
#                 'Model': model_type,
#                 'Spearman': row['Spearman']
#             })
#
#     results_df = pd.DataFrame(results)
#     results_df.to_csv('./analysis/Spearman_detailed.csv', index=False)
#

if __name__ == '__main__':
    reference_csv = './data/research_data/test_full.csv'

    predicts = ['classical_300_layer3', 'qh8_300_layer3', 'ensemble_qh8_300_layer3',
               'classical_1200_layer3', 'qh8_1200_layer3', 'ensemble_qh8_1200_layer3',
               'classical_2100_layer3', 'qh2_2100_layer3', 'ensemble_qh2_2100_layer3',
               'classical_300_layer2', 'qh2_300_layer2', 'ensemble_qh2_300_layer2',
               'classical_1200_layer2', 'qh8_1200_layer2', 'ensemble_qh8_1200_layer2',
               'classical_2100_layer2', 'qh2_2100_layer2', 'ensemble_qh2_2100_layer2',
               'classical_2048_layer1', 'qh2_2048_layer1', 'ensemble_qh2_2048_layer1']

    results = []
    for idx, predict in enumerate(predicts):
        group_id = f'{idx // 3 + 1}'
        predict_csv = './output/model/' + predict + '/fold_0/' + predict + '.csv'

        tmse_df = calculate_tmse(reference_csv, predict_csv)
        mean_tmse = tmse_df['TMSE'].mean()

        spearman_df = calculate_spearman(reference_csv, predict_csv)
        mean_spearman = spearman_df['Spearman'].mean()
        # q1_spearman = spearman_df['Spearman'].quantile(0.25)
        # median_spearman = spearman_df['Spearman'].quantile(0.5)
        # q3_spearman = spearman_df['Spearman'].quantile(0.75)

        pearson_df = calculate_pearson(reference_csv, predict_csv)
        mean_pearson = pearson_df['Pearson'].mean()
        # q1_pearson = pearson_df['Pearson'].quantile(0.25)
        # median_pearson = pearson_df['Pearson'].quantile(0.5)
        # q3_pearson = pearson_df['Pearson'].quantile(0.75)

        results.append({
            'Group': group_id,
            'Model': predict,
            'tmse': mean_tmse,
            'Spearman Average': mean_spearman,
            # 'Spearman Q1': q1_spearman,
            # 'Spearman Median': median_spearman,
            # 'Spearman Q3': q3_spearman,
            'Pearson Average': mean_pearson,
            # 'Pearson Q1': q1_pearson,
            # 'Pearson Median': median_pearson,
            # 'Pearson Q3': q3_pearson
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('./analysis/metrics_tmse_summary.csv', index=False)

