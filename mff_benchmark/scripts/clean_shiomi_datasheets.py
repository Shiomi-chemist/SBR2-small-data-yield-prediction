"""Clean the four data sheets from S. Shiomi (data/raw/MFF_*.xlsx) -> data/clean/*.csv

Fixes applied (all logged):
  * drop fully empty trailing rows (deamino has one)
  * drop empty placeholder columns named 'none', 'none.1'
  * UmAS solvent column: 'C1CCCO2.[H]O[H]' ... 'C1CCCO41.[H]O[H]' are Excel auto-increment
    corruptions of THF/water 'C1CCCO1.[H]O[H]' -> restored
  * every SMILES is checked with RDKit; the script fails loudly on anything else invalid
"""
import re, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw"
DST = ROOT / "data/clean"
DST.mkdir(exist_ok=True)

# which SMILES columns are *substrates* (what SBR uses) vs reagents/conditions
SUBSTRATES = {
    "redam":   ["aldehyde_smiles", "amine_smiles"],
    "electro": ["arylbromide_smiles", "alcohol_smiles"],
    "deamino": ["amine_smiles", "BA_smiles"],
    "UmAS":    ["amine_smiles", "amine_smiles.1"],   # alpha-bromonitroalkane + amine
}


def main():
    for name in ["redam", "electro", "deamino", "UmAS"]:
        df = pd.read_excel(SRC / f"MFF_{name}.xlsx", sheet_name="data")
        n0 = len(df)
        df = df.drop(columns=[c for c in df.columns if c.startswith("none")])
        body = df.drop(columns=["reaction_id", "dataset"])
        df = df[~body.isna().all(axis=1)].reset_index(drop=True)
        smi_cols = [c for c in df.columns if "smiles" in c]
        if name == "UmAS":
            bad = df["solvent_smiles"].str.match(r"^C1CCCO\d+\.", na=False)
            df.loc[bad, "solvent_smiles"] = df.loc[bad, "solvent_smiles"].str.replace(r"^C1CCCO\d+\.", "C1CCCO1.", regex=True)
            print(f"[{name}] repaired {int(bad.sum())} corrupted THF/water solvent SMILES")
        for c in smi_cols:
            for v in df[c].dropna().astype(str).str.strip().unique():
                if Chem.MolFromSmiles(v) is None:
                    sys.exit(f"[{name}] invalid SMILES in {c}: {v}")
        df["yield"] = pd.to_numeric(df["yield"])
        assert df["reaction_id"].is_unique
        df.to_csv(DST / f"{name}.csv", index=False)
        reagents = [c for c in smi_cols if c not in SUBSTRATES[name]]
        print(f"[{name}] rows {n0} -> {len(df)}; substrates={SUBSTRATES[name]}; reagent cols={reagents}; "
              f"yield {df['yield'].min():.0f}-{df['yield'].max():.0f} (mean {df['yield'].mean():.1f}, sd {df['yield'].std():.1f})")


if __name__ == "__main__":
    main()
