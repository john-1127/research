from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols


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
    "alkyl_Cl_Br_CX": "[CX4][Cl,Br]",
    "Si_O": "SiO",
    "Si=O": "Si=O",
    "C_Si": "[#6]Si",
    "P_O": "PO",
    "P=O": "P=O",
    "C_P": "[#6]P",
}


def add_Hs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_with_H = Chem.AddHs(mol)
    new_smiles = Chem.MolToSmiles(mol_with_H)

    return new_smiles


def functional_count(smiles, substruct_smiles):
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(substruct_smiles)
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


def cosine_similarity(smiles1, smiles2):
    vec1 = count_all_functional_groups(smiles1)
    vec2 = count_all_functional_groups(smiles2)
    process_vec1 = preprocessing(vec1)
    process_vec2 = preprocessing(vec2)
    v1 = np.array(process_vec1)
    v2 = np.array(process_vec2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == '__main__':
    first = 'Cc1cc(C)c(F)c(C)c1'
    vector = count_all_functional_groups(first)
    vector = preprocessing(vector)
    print(vector)
