"""Code S1. Generate four SBR feature matrices from reaction SMILES.
 
Input CSV (one row per reaction): num, reaction_id, sm1, sm2, prod, yield.
Outputs retain row order and metadata; Morgan fingerprints use radius 2,
2048 bits, and useChirality=False. Invalid structures raise an error.
 
SBR-C2  = [FP(A), FP(B)]           (4096 features)
SBR-2   = FP(A) + FP(B)           (2048 features)
SBR-2P  = [FP(A) + FP(B), FP(P)]  (4096 features)
SBR-3   = FP(A) + FP(B) + FP(P)  (2048 features)
 
"+" denotes element-wise arithmetic addition, not bitwise OR.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
 
# Edit only these settings to use a different reaction dataset.
# INPUT_CSV = Path("data/ML_redam.csv")
# INPUT_CSV = Path("data/ML_deamino.csv")
# INPUT_CSV = Path("data/ML_electro.csv")
INPUT_CSV = Path("data/ML_UmAS.csv")
OUTPUT_DIR = Path("SBR_features")
RADIUS = 2
N_BITS = 2048
META = ["num", "reaction_id", "sm1", "sm2", "prod", "yield"]
 
 
def fingerprints(smiles, generator, field, row_number):
    if pd.isna(smiles) or not str(smiles).strip():
        raise ValueError(f"Row {row_number}: missing {field}")
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError(f"Row {row_number}: invalid {field}: {smiles}")
    fp = generator.GetFingerprint(molecule)
    values = np.zeros(N_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, values)
    return values
 
 
def main():
    df = pd.read_csv(INPUT_CSV)
    missing = [col for col in META if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if df["num"].isna().any() or df["num"].duplicated().any():
        raise ValueError("The 'num' column must contain unique, nonmissing IDs")
    if not np.isfinite(pd.to_numeric(df["yield"], errors="coerce")).all():
        raise ValueError("Non-numeric or missing reaction yields detected")
 
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS, includeChirality=False
    )
    a, b, p = [], [], []
    for i, (_, record) in enumerate(df.iterrows(), start=2):
        a.append(fingerprints(record["sm1"], gen, "sm1", i))
        b.append(fingerprints(record["sm2"], gen, "sm2", i))
        p.append(fingerprints(record["prod"], gen, "prod", i))
 
    a, b, p = np.stack(a), np.stack(b), np.stack(p)
    ab = a.astype(np.uint8) + b.astype(np.uint8)
    abp = ab + p.astype(np.uint8)
 
    feature_sets = {
        "SBR-C2": (np.concatenate([a, b], axis=1),
                    [f"Morgan_A_{j}" for j in range(N_BITS)] +
                    [f"Morgan_B_{j}" for j in range(N_BITS)]),
        "SBR-2": (ab, [f"Morgan_AB_{j}" for j in range(N_BITS)]),
        "SBR-2P": (np.concatenate([ab, p], axis=1),
                     [f"Morgan_AB_{j}" for j in range(N_BITS)] +
                     [f"Morgan_P_{j}" for j in range(N_BITS)]),
        "SBR-3": (abp, [f"Morgan_ABP_{j}" for j in range(N_BITS)]),
    }
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (matrix, columns) in feature_sets.items():
        output = pd.concat(
            [df[META].reset_index(drop=True), pd.DataFrame(matrix, columns=columns)],
            axis=1,
        )
        target = OUTPUT_DIR / f"{INPUT_CSV.stem}_{name}.csv"
        output.to_csv(target, index=False)
        print(f"{name}: {len(output)} reactions, {len(columns)} features -> {target}")
 
 
if __name__ == "__main__":
    main()

