import pandas as pd
from rdkit import Chem
import argparse

# -----------------------------
#=       Functions
# -----------------------------

def contains_only_CHO_N(mol):
    allowed_atoms = {"C", "H", "O", "N"}
    return all(atom.GetSymbol() in allowed_atoms for atom in mol.GetAtoms())

def is_single_component(mol):
    return len(Chem.GetMolFrags(mol)) == 1

def has_no_radicals(mol):
    return not any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())

def remove_isotopes(mol):
    return all(atom.GetIsotope() == 0 for atom in mol.GetAtoms())

def carbon_count(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

# SMARTS patterns
primary_amine   = Chem.MolFromSmarts("[N;X3;H2;D1;+0;!$(NC=O)]")
secondary_amine = Chem.MolFromSmarts("[N;X3;H1;D2;+0;!$(NC=O)]")
tertiary_amine  = Chem.MolFromSmarts("[N;X3;H0;D3;+0;!$(NC=O);!$([N+])]")

def classify_amine_smarts(mol):
    if mol.HasSubstructMatch(primary_amine):
        return "Primary"
    if mol.HasSubstructMatch(secondary_amine):
        return "Secondary"
    if mol.HasSubstructMatch(tertiary_amine):
        return "Tertiary"
    return None


# -----------------------------
# Main Function
# -----------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Filter ChEMBL dataset for CHON amines (≤12C) with pKa constraints."
    )

    parser.add_argument(
        "input_file",
        help="Input CSV file (e.g., chembl_36.csv)"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="filtered_output.csv",
        help="Output CSV file name"
    )

    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.input_file)

    # 1. Remove missing SMILES
    df = df.dropna(subset=["Smiles"])
    df = df[df["Smiles"].str.strip() != ""]

    # 2. Remove missing CX Basic pKa
    df = df.dropna(subset=["CX Basic pKa"])

    valid_rows = []
    amine_classes = []

    print("Applying filters...")

    for _, row in df.iterrows():
        smiles = row["Smiles"]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        if not contains_only_CHO_N(mol):
            continue

        if not is_single_component(mol):
            continue

        if not has_no_radicals(mol):
            continue

        if not remove_isotopes(mol):
            continue

        if carbon_count(mol) > 12:
            continue

        amine_type = classify_amine_smarts(mol)
        if amine_type is None:
            continue

        # Zheng et al. filter
        acidic_pka = row.get("CX Acidic pKa")
        basic_pka = row.get("CX Basic pKa")

        if pd.notna(acidic_pka):
            if acidic_pka < basic_pka:
                continue

        valid_rows.append(row)
        amine_classes.append(amine_type)

    filtered_df = pd.DataFrame(valid_rows)
    filtered_df["Amine Class"] = amine_classes

    filtered_df.to_csv(args.output, index=False)

    print("Filtering complete.")
    print(f"Final dataset size: {len(filtered_df)} molecules")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
