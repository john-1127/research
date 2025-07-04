import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

def generate_molecular_structure(smiles: str, save_img_path: str):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule:
        img = Draw.MolToImage(molecule, size=(300, 300))
        img.save(save_img_path)
    else:
        print("Fail to generate molecular structure,")


def generate_spectra_comparison(
    wavenumbers: list, lines: list, save_img_path: str, start=50, interval=250
):
    plt.figure(figsize=(12, 4))
    for line in lines:
        
        plt.plot(
            wavenumbers,
            line["absorbance"],
            label=line["label"],
            color=line.get("color", "gray"),
            linestyle=line.get("linestyle", "-"),
            linewidth=line.get("linewidth", 2),
        )
    xticks = wavenumbers[start::interval]
    plt.xticks(xticks, fontname="sans serif", fontweight="bold")
    plt.xlim(0, 1800)
    plt.gca().invert_xaxis()
    plt.gca().set_yticks([])

    plt.xlabel(
        "Wavenumbers (cm⁻¹)", fontname="sans serif", fontsize=12, fontweight="bold"
    )
    plt.ylabel(
        "Absorbance (a.u.)", fontname="sans serif", fontsize=12, fontweight="bold"
    )
    plt.title("Absorbance Spectrum", fontname="sans serif", fontweight="bold")
    plt.legend(frameon=False, loc='upper left')
    plt.grid(False)

    output_file = save_img_path
    plt.savefig(output_file, dpi=300)
    plt.close()

def generate_spectra_comparison_test(
    wavenumbers: list, lines: list, save_img_path: str, start=50, interval=250
):
    plt.figure(figsize=(12, 4))
    for line in lines:
        x = np.array(wavenumbers, dtype=float) 
        y = np.array(line["absorbance"], dtype=float)
        mask = ~np.isnan(y)
        x_plot = x[mask]
        y_plot = y[mask]

        plt.plot(
            x_plot,
            y_plot,
            label=line["label"],
            color=line.get("color", "gray"),
            linestyle=line.get("linestyle", "-"),
            linewidth=line.get("linewidth", 2),
        )
    xticks = x[start::interval]
    plt.xticks(xticks, fontname="sans serif", fontweight="bold")
    plt.xlim(x[0], x[-1])
    plt.gca().invert_xaxis()
    plt.gca().set_yticks([])

    plt.xlabel(
        "Wavenumbers (cm⁻¹)", fontname="sans serif", fontsize=12, fontweight="bold"
    )
    plt.ylabel(
        "Absorbance (a.u.)", fontname="sans serif", fontsize=12, fontweight="bold"
    )
    plt.title("Absorbance Spectrum", fontname="sans serif", fontweight="bold")
    plt.legend(frameon=False, loc='upper left')
    plt.grid(False)

    output_file = save_img_path
    plt.savefig(output_file, dpi=300)
    plt.close()


def generate_qnn_architecture():
    import pennylane as qml
    import matplotlib.pyplot as plt
    import torch
    n_qubits= 4
    n_layers = 2

    # shape (L, M, 3) for stronglyEntanglingLaters
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def qnode(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    params = torch.rand(shape)
    inputs = torch.rand(16) 
    fig, ax = qml.draw_mpl(qnode, level='device')(inputs, params)
    fig.savefig("./circuit.jpg", format="jpg", dpi=300, bbox_inches='tight')

# example

# file_path_1 = "./data/research_data/test_full.csv"
file_path_1 = "./nist/test_spectra.csv"
df_1 = pd.read_csv(file_path_1)


# file_path_2 = "./output/model/ensemble_qh2_2100_layer3/fold_0/ensemble_qh2_2100_layer3.csv"
file_path_2 = "./nist/predict.csv"
df_2 = pd.read_csv(file_path_2)

wavenumbers = [col for col in df_1.columns if col != "smiles" and col != "epi_unc"]

lines = [
    {"absorbance": df_1.loc[0, wavenumbers], "label": "Spectrum"},
    {
        "absorbance": df_2.loc[0, wavenumbers],
        "label": "Model Predict Spectrum",
        "color": "blue",
        "linestyle": (0, (2, 2)),
        "linewidth": 1.5,
    },
]
smiles = df_1.loc[0, 'smiles']
print(f"Selected SMILES: {smiles}")
smiles = df_2.loc[0, 'smiles']
print(f"Selected SMILES: {smiles}")

# bad1 = 'CN(c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-])[N+](=O)[O-]'
# bad2 = '[2H]C1([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C1([2H])[2H]'
# bad3 = 'C=[N+]([O-])[N+](=O)[O-]'
# bad4 = 'O=C1OC(=C2OC(=O)O2)O1'
# bad5 = 'c1ccccc1'
# diff_ex1 = 'CC#CC1(O)CCCC(OCCN(C(C)C)C(C)C)C1'
# diff_ex2 = 'c1cnc2ccc(C3CCC3CNC3CC3)cc2c1'
# diff_ensemble_ex1 = 'CCC1=C(C)CCC1'
generate_spectra_comparison_test(wavenumbers, lines, "./0.png")
# generate_molecular_structure('CCC(CC)N1C(=Nc2cccc(Cl)c2Cl)SCC1CC(C)C', './output/figures/compare2model/FFNN/5.png')


# file_path_1 = "./data/research_data/test_full.csv"
# df_1 = pd.read_csv(file_path_1)
#
#
# file_path_2 = "./output/model/classical_2100_layer3/fold_0/classical_2100_layer3.csv"
# df_2 = pd.read_csv(file_path_2)
#
# file_path_3 = "./output/model/qh2_2100_layer3/fold_0/qh2_2100_layer3.csv"
# df_3 = pd.read_csv(file_path_3)
#
# file_path_4 = "./output/model/ensemble_qh2_2100_layer3/fold_0/ensemble_qh2_2100_layer3.csv"
# df_4 = pd.read_csv(file_path_4)
#
# wavenumbers = [col for col in df_1.columns if col != "smiles" and col != "epi_unc"]
#
#
# lines = [
#     {"absorbance": df_1.loc[6016, wavenumbers], "label": "Reference Spectrum"},
#     {
#         "absorbance": df_2.loc[6016, wavenumbers],
#         "label": "FFNN Model Predict Spectrum",
#         "color": "blue",
#         "linestyle": (0, (2, 2)),
#         "linewidth": 1.5,
#     },
#     {
#         "absorbance": df_3.loc[6016, wavenumbers],
#         "label": "HQNN Model Predict Spectrum",
#         "color": "red",
#         "linestyle": (0, (2, 2)),
#         "linewidth": 1.5,
#     },
#     {
#         "absorbance": df_4.loc[6016, wavenumbers],
#         "label": "Ensemble Model Predict Spectrum",
#         "color": "green",
#         "linewidth": 1.5,
#     },
#
# ]
# smiles = df_1.loc[6016, 'smiles']
# print(f"Selected SMILES: {smiles}")
# smiles = df_2.loc[6016, 'smiles']
# print(f"Selected SMILES: {smiles}")
#
# bad1 = 'CN(c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-])[N+](=O)[O-]'
# bad2 = '[2H]C1([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C1([2H])[2H]'
# bad3 = 'C=[N+]([O-])[N+](=O)[O-]'
# bad4 = 'O=C1OC(=C2OC(=O)O2)O1'
# diff_ex1 = 'CC#CC1(O)CCCC(OCCN(C(C)C)C(C)C)C1'
# generate_spectra_comparison(wavenumbers, lines, "./output/figures/examples.png")

