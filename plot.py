import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw


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
    plt.legend()
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
# file_path_1 = "./output/model/hybrid.csv"
# df_1 = pd.read_csv(file_path_1)
#
# file_path_2 = "./data/computed_data/test_full.csv"
# df_2 = pd.read_csv(file_path_2)
#
# wavenumbers = [col for col in df_1.columns if col != "smiles" and col != "epi_unc"]
#
#
# lines = [
#     {"absorbance": df_1.loc[0, wavenumbers], "label": "GFN2-xTB"},
#     {
#         "absorbance": df_2.loc[0, wavenumbers],
#         "label": "Model(GFN2-xTB)",
#         "color": "blue",
#         "linestyle": (0, (2, 2)),
#         "linewidth": 1.5,
#     },
# ]
# generate_spectra_comparison(wavenumbers, lines, "./output/figure/hybrid.png")

generate_qnn_architecture()
