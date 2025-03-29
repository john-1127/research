import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def find_low_score_samples(file_path, threshold=0.85):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    print(f"\nFile: {file_path}")
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue

        smiles, score_str = parts
        try:
            score = float(score_str)
            if score < threshold:
                print(f"Sample {i}: SMILES={smiles}, SIS={score}")
        except ValueError:
            continue


def read_scores_below_threshold(file_path, threshold):
    indices = []
    scores = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            try:
                score = float(parts[1])
                if score < threshold:
                    indices.append(i)
                    scores.append(score)
            except ValueError:
                continue
    return indices, scores

file_1 = "./output/sis/sis_morgan_classical.txt"
file_2 = "./output/sis/sis_morgan_hybrid_probs.txt"


threshold = 0.5


def compare_scores_if_classical_low(file_classical, file_hybrid, threshold=0.6):
    with open(file_classical, 'r') as f1, open(file_hybrid, 'r') as f2:
        lines_classical = f1.readlines()
        lines_hybrid = f2.readlines()

    diffs = []
    smiles_list = []

    for i, (line_c, line_h) in enumerate(zip(lines_classical, lines_hybrid)):
        parts_c = line_c.strip().split('\t')
        parts_h = line_h.strip().split('\t')

        if len(parts_c) != 2 or len(parts_h) != 2:
            continue

        smiles_c, score_c_str = parts_c
        smiles_h, score_h_str = parts_h

        try:
            score_c = float(score_c_str)
            score_h = float(score_h_str)
        except ValueError:
            continue

        if score_c < threshold:
            diff = score_h - score_c
            print(f"Sample {i}: SMILES={smiles_c}")
            print(f"  Classical={score_c:.4f}, Hybrid={score_h:.4f}, Diff={diff:+.4f}")

            diffs.append(diff)
            smiles_list.append(smiles_c)

    if diffs:
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(diffs)), diffs, marker='o', color='purple')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Sample (low classical score)")
        plt.ylabel("Hybrid - Classical (Δ Score)")
        plt.title(f"Score Difference (Hybrid - Classical) where Classical < {threshold}")
        plt.tight_layout()
        plt.savefig("./output/sis/sis_comparison_6.jpg", dpi=300)
        plt.close()


def plot_residual_heatmap(true_csv, pred_csv, vmin=-0.03, vmax=0.03):
    df_true = pd.read_csv(true_csv)
    df_pred = pd.read_csv(pred_csv)

    y_true = df_true.iloc[:300, 51:501].values
    y_pred = df_pred.iloc[:300, 51:501].values

    residual = y_true - y_pred
    print("Residual min:", residual.min(), "max:", residual.max())
    plt.figure(figsize=(14, 6))
    sns.heatmap(residual, cmap="coolwarm", center=0, vmax=vmax)
    plt.xlabel("Wavelength Index")
    plt.ylabel("Sample Index")
    plt.title("Residual Heatmap (True - Predicted)")
    plt.tight_layout()

    output_path = "./output/sis/residual_heatmap.jpg"
    plt.savefig(output_path)
    plt.close()

    print(f"Residual heatmap saved as {output_path}")

plot_residual_heatmap("./data/research_data/test_full.csv", "./output/model/morgan_classical/fold_0/classical.csv")
