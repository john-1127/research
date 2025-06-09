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
        plt.plot(range(len(diffs)), diffs, marker='o', markersize=3, color='purple')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel("Sample (low classical score)")
        plt.ylabel("Hybrid - Classical (Δ SIS)")
        plt.title(f"SIS Score Difference (Hybrid - Classical) where Classical < {threshold}")
        plt.tight_layout()
        plt.savefig("./output/sis/sis_comparison_ensemble_vs_classical.jpg", dpi=300)
        plt.close()

def compare_scores_if_classical_low_log(file_classical, file_hybrid, output_txt, threshold=0.7):
    with open(file_classical, 'r') as f1, open(file_hybrid, 'r') as f2:
        lines_classical = f1.readlines()
        lines_hybrid = f2.readlines()

    diffs = []
    smiles_list = []
    warnings = []
    high_diff_cases = []

    with open(output_txt, 'w') as fout:
        fout.write(f"# Comparison Report (threshold = {threshold})\n")
        fout.write(f"# Format: SMILES, Classical, Hybrid, Diff\n\n")

        for i, (line_c, line_h) in enumerate(zip(lines_classical, lines_hybrid), start=1):
            parts_c = line_c.strip().split('\t')
            parts_h = line_h.strip().split('\t')

            if len(parts_c) != 2 or len(parts_h) != 2:
                warnings.append(f"[Line {i}] Invalid format")
                continue

            smiles_c, score_c_str = parts_c
            smiles_h, score_h_str = parts_h

            try:
                score_c = float(score_c_str)
                score_h = float(score_h_str)
            except ValueError:
                warnings.append(f"[Line {i}] ValueError: cannot parse SIS score → {parts_c}, {parts_h}")
                continue

            if score_c < threshold:
                diff = score_h - score_c
                fout.write(f"{smiles_c}\t{score_c:.4f}\t{score_h:.4f}\t{diff:+.4f}\n")
                diffs.append(diff)
                smiles_list.append(smiles_c)

                if diff >= 0.15:
                    high_diff_cases.append((i, smiles_c, diff))

        if high_diff_cases:
            fout.write("\n# High Diff Cases (Diff ≥ 0.15)\n")
            for line_no, smiles, diff in high_diff_cases:
                fout.write(f"Line {line_no}: {smiles} (Δ = {diff:+.4f})\n")

        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            fout.write(f"\n# Mean Δ (Hybrid - Classical): {mean_diff:+.4f}\n")

        if warnings:
            fout.write("\n# Warnings / Invalid lines\n")
            for w in warnings:
                fout.write(w + "\n")

    print(f"Comparison complete. Results written to {output_txt}")

def compare_scores_ensemble_vs_classical_and_hybrid(file_classical, file_hybrid, file_ensemble, output_txt, threshold=0.05):
    with open(file_classical, 'r') as f1, open(file_hybrid, 'r') as f2, open(file_ensemble, 'r') as f3:
        lines_classical = f1.readlines()
        lines_hybrid = f2.readlines()
        lines_ensemble = f3.readlines()

    all_results = []
    high_diff_cases = []

    with open(output_txt, 'w') as fout:
        fout.write(f"# Ensemble Comparison Report (threshold = {threshold})\n")
        fout.write("# Format: SMILES\tClassical\tHybrid\tEnsemble\t(Ensemble - Classical)\t(Ensemble - Hybrid)\n")

        for i, (lc, lh, le) in enumerate(zip(lines_classical, lines_hybrid, lines_ensemble), start=1):
            parts_c = lc.strip().split('\t')
            parts_h = lh.strip().split('\t')
            parts_e = le.strip().split('\t')

            if len(parts_c) != 2 or len(parts_h) != 2 or len(parts_e) != 2:
                continue

            smiles_c, score_c_str = parts_c
            smiles_h, score_h_str = parts_h
            smiles_e, score_e_str = parts_e

            try:
                score_c = float(score_c_str)
                score_h = float(score_h_str)
                score_e = float(score_e_str)
            except ValueError:
                continue

            diff_ec = score_e - score_c
            diff_eh = score_e - score_h
            fout.write(f"{smiles_c}\t{score_c:.4f}\t{score_h:.4f}\t{score_e:.4f}\t{diff_ec:+.4f}\t{diff_eh:+.4f}\n")
            all_results.append((i, smiles_c, score_c, score_h, score_e, diff_ec, diff_eh))

            if diff_ec >= threshold and diff_eh >= threshold:
                high_diff_cases.append((i, smiles_c, diff_ec, diff_eh))

        if high_diff_cases:
            fout.write(f"\n# High Δ SIS Cases (Δ ≥ {threshold} for both comparisons)\n")
            for idx, smiles, d_ec, d_eh in high_diff_cases:
                fout.write(f"Line {idx}: {smiles} (Δ EC = {d_ec:+.4f}, Δ EH = {d_eh:+.4f})\n")

    print(f"Ensemble comparison completed. Results written to {output_txt}")
# def plot_residual_heatmap(true_csv, pred_csv, vmin=-0.03, vmax=0.03):
#     df_true = pd.read_csv(true_csv)
#     df_pred = pd.read_csv(pred_csv)
#
#     y_true = df_true.iloc[:300, 51:501].values
#     y_pred = df_pred.iloc[:300, 51:501].values
#
#     residual = y_true - y_pred
#     print("Residual min:", residual.min(), "max:", residual.max())
#     plt.figure(figsize=(14, 6))
#     sns.heatmap(residual, cmap="coolwarm", center=0, vmax=vmax)
#     plt.xlabel("Wavelength Index")
#     plt.ylabel("Sample Index")
#     plt.title("Residual Heatmap (True - Predicted)")
#     plt.tight_layout()
#
#     output_path = "./output/sis/residual_heatmap.jpg"
#     plt.savefig(output_path)
#     plt.close()
#
#     print(f"Residual heatmap saved as {output_path}")
#
# plot_residual_heatmap("./data/research_data/test_full.csv", "./output/model/morgan_classical/fold_0/classical.csv")
if __name__ == '__main__':
    file_1 = "./output/sis/classical_2100_layer3.txt"
    file_2 = "./output/sis/qh2_2100_layer3.txt"
    file_3 = "./output/sis/ensemble_qh2_2100_layer3.txt"
    output_path = "./output/sis/diff_ensemble_vs_classical_log.txt"
    compare_scores_if_classical_low_log(file_1, file_3, output_path, threshold=0.7)
    compare_scores_if_classical_low(file_1, file_3, threshold=0.7)
    # compare_scores_ensemble_vs_classical_and_hybrid(file_1, file_2, file_3, output_path, threshold=0.02)
