"""MFF encoder: 24 RDKit fingerprints per molecule, concatenated per reaction component.

The list of fingerprints and their parameters is copied 1:1 from
EasyChemML/Encoder/MFF.py (mkueh/EasyChemML):

  * RDKit topological fp, branched, maxPath 2/4/6/8          (4)
  * RDKit topological fp, linear,   maxPath 2/4/6/8          (4)
  * Morgan (ECFP-like), radius 0/2/4/6                        (4)
  * Morgan with feature invariants (FCFP-like), radius 0/2/4/6 (4)
  * Layered fp, maxPath 2/4/6/8                               (4)
  * Avalon                                                    (1)
  * MACCS keys (167 bits, fixed)                              (1)
  * Hashed atom pairs                                         (1)
  * Hashed topological torsions                               (1)

Every fingerprint except MACCS has ``fp_length`` bits. A missing / unparsable
component gives an all-zero vector (EasyChemML writes 'NA' and emits zeros).
"""
from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# (name, kind, kwargs) in the exact order used by EasyChemML MFF.convert()
FP_SPECS: List[tuple] = (
    [("rdkit_branched_p%d" % p, "rdkit", {"maxPath": p}) for p in (2, 4, 6, 8)]
    + [("rdkit_linear_p%d" % p, "rdkit", {"maxPath": p, "branchedPaths": False}) for p in (2, 4, 6, 8)]
    + [("morgan_r%d" % r, "morgan", {"radius": r}) for r in (0, 2, 4, 6)]
    + [("morgan_feat_r%d" % r, "morgan", {"radius": r, "useFeatures": True}) for r in (0, 2, 4, 6)]
    + [("layered_p%d" % p, "layered", {"maxPath": p}) for p in (2, 4, 6, 8)]
    + [("avalon", "avalon", {}), ("maccs", "maccs", {}), ("atom_pairs", "atom_pairs", {}),
       ("topological_torsions", "topological_torsions", {})]
)
MACCS_LEN = 167


def _bv_to_array(bv) -> np.ndarray:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.int8, copy=False)


def _single_fp(mol, kind: str, length: int, kwargs: dict) -> np.ndarray:
    """Identical RDKit calls to EasyChemML FingerprintGenerator (deprecation warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if kind == "rdkit":
            bv = Chem.RDKFingerprint(mol=mol, fpSize=length, **kwargs)
        elif kind == "morgan":
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=length, **kwargs)
        elif kind == "layered":
            bv = Chem.LayeredFingerprint(mol, fpSize=length, **kwargs)
        elif kind == "avalon":
            bv = pyAvalonTools.GetAvalonFP(mol, nBits=length, **kwargs)
        elif kind == "maccs":
            bv = MACCSkeys.GenMACCSKeys(mol, **kwargs)
        elif kind == "atom_pairs":
            bv = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=length, **kwargs)
        elif kind == "topological_torsions":
            bv = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=length, **kwargs)
        else:
            raise ValueError(kind)
    return _bv_to_array(bv)


def is_missing(smiles) -> bool:
    if smiles is None:
        return True
    if isinstance(smiles, float) and np.isnan(smiles):
        return True
    s = str(smiles).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null", "-"}


class MFFEncoder:
    """Encode SMILES -> MFF bit vectors.

    Parameters
    ----------
    fp_length : int
        Bits per fingerprint (all but MACCS, which is RDKit's fixed 167-bit key set;
        the paper counts it as 166 because bit 0 is unused). Sandfort et al. 2020
        used 3096 bits -> 23*3096 + 166 = 71,374 bits per molecule (SI section 1.2).
    """

    def __init__(self, fp_length: int = 3096):
        self.fp_length = int(fp_length)
        self.n_bits_per_mol = (len(FP_SPECS) - 1) * self.fp_length + MACCS_LEN

    # ---- per molecule ---------------------------------------------------
    def feature_names_one(self, prefix: str = "") -> List[str]:
        names: List[str] = []
        for name, kind, _ in FP_SPECS:
            n = MACCS_LEN if kind == "maccs" else self.fp_length
            names.extend(f"{prefix}{name}_{i}" for i in range(n))
        return names

    def encode_smiles(self, smiles) -> np.ndarray:
        if is_missing(smiles):
            return np.zeros(self.n_bits_per_mol, dtype=np.int8)
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        parts = [_single_fp(mol, kind, self.fp_length, kw) for _, kind, kw in FP_SPECS]
        return np.concatenate(parts).astype(np.int8, copy=False)

    # ---- per reaction (several components) ------------------------------
    def encode_table(self, smiles_columns: Sequence[Iterable], on_error: str = "raise") -> np.ndarray:
        """smiles_columns: list of per-component iterables (each len = n_reactions).

        Returns int8 array (n_reactions, n_components * n_bits_per_mol).
        Unique SMILES are encoded once and cached (the same reagents recur many times).
        """
        cols = [list(c) for c in smiles_columns]
        n = len(cols[0])
        cache: dict = {}
        out = np.zeros((n, len(cols) * self.n_bits_per_mol), dtype=np.int8)
        for j, col in enumerate(cols):
            lo, hi = j * self.n_bits_per_mol, (j + 1) * self.n_bits_per_mol
            for i, smi in enumerate(col):
                key = "" if is_missing(smi) else str(smi).strip()
                if key not in cache:
                    try:
                        cache[key] = self.encode_smiles(key)
                    except ValueError:
                        if on_error == "raise":
                            raise
                        cache[key] = np.zeros(self.n_bits_per_mol, dtype=np.int8)
                out[i, lo:hi] = cache[key]
        return out

    def feature_names(self, component_names: Sequence[str]) -> List[str]:
        names: List[str] = []
        for c in component_names:
            names.extend(self.feature_names_one(prefix=f"{c}__"))
        return names
