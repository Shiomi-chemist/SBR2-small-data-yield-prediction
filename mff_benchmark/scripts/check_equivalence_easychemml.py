"""Bit-level comparison of mff.encoder against the reference EasyChemML FingerprintGenerator.

Usage:
  git clone --depth 1 https://github.com/mkueh/EasyChemML /path/to/EasyChemML
  python scripts/check_equivalence_easychemml.py /path/to/EasyChemML [fp_length]

Only the pure-python FingerprintGenerator file is loaded (no EasyChemML install needed).
"""
import sys, warnings, re, types, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
src = Path(sys.argv[1]) / "EasyChemML/Encoder/impl_FingerprintEncoder/FingerprintGenerator.py"
code = "from __future__ import annotations\n" + re.sub(r"^(from EasyChemML.*|if TYPE_CHECKING:\n.*)$", "", src.read_text(), flags=re.M)
import enum
class BatchDatatypClass(enum.Enum):
    NUMPY_INT8 = 'int8'; NUMPY_INT32 = 'int32'
    def to_numpy(self): return np.dtype(self.value)
    @staticmethod
    def get_dtype_lvl(x): return 0 if x == BatchDatatypClass.NUMPY_INT8 else 1
    @staticmethod
    def get_by_lvl(l): return BatchDatatypClass.NUMPY_INT8 if l == 0 else BatchDatatypClass.NUMPY_INT32
fpgen_ref = types.ModuleType('fpgen_ref'); fpgen_ref.BatchDatatypClass = BatchDatatypClass; exec(code, fpgen_ref.__dict__)
from rdkit import Chem
from mff.encoder import MFFEncoder
names = [('rdkit',BatchDatatypClass.NUMPY_INT8,'var')]*8 + [('morgan_circular',BatchDatatypClass.NUMPY_INT8,'var')]*8 \
      + [('layerdfingerprint',BatchDatatypClass.NUMPY_INT8,'var')]*4 + [('avalon',BatchDatatypClass.NUMPY_INT8,'var'),
        ('maccs',BatchDatatypClass.NUMPY_INT8,'167'),('atom_pairs',BatchDatatypClass.NUMPY_INT8,'var'),('topological_torsions',BatchDatatypClass.NUMPY_INT8,'var')]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
args = [{'length':L,'maxPath':p} for p in (2,4,6,8)] + [{'length':L,'maxPath':p,'branchedPaths':False} for p in (2,4,6,8)] \
     + [{'length':L,'radius':r} for r in (0,2,4,6)] + [{'length':L,'radius':r,'useFeatures':True} for r in (0,2,4,6)] \
     + [{'length':L,'maxPath':p} for p in (2,4,6,8)] + [{'length':L},{},{'length':L},{'length':L}]
smiles = ['COc1ccc(Br)cc1','CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1','O=C([O-])[O-].[Cs+].[Cs+]','CC(=O)O[Pd]OC(C)=O','C1COCCN1','c1ccc2ccccc2c1','CCCC[N+](CCCC)(CCCC)CCCC.[Br-]']
G = fpgen_ref.FingerprintGenerator()
enc = MFFEncoder(L)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ref = G.generateArrOfFingerprints([Chem.MolFromSmiles(s) for s in smiles], names, args)
mine = np.stack([enc.encode_smiles(s) for s in smiles])
print(f'L={L}: ref {ref.shape} mine {mine.shape} identical: {ref.shape == mine.shape and bool((ref == mine).all())}  bits set: {int(mine.sum())}')
