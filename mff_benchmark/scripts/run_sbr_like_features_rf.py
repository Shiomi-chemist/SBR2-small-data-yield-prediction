"""Control experiment: SBR1.5-like features (Morgan count fingerprint, radius 2, 2048 bits, summed over the
two substrates) with the MFF paper's Random Forest, on the same 70/30 splits. Separates the effect of the
regressor (RF vs ElasticNet) from the effect of the representation (MFF vs Morgan).
Encoding confirmed by S. Shiomi (2026-10-21): Morgan radius 2, 2048 bits, count encoding, summed over the two substrates."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.clean_shiomi_datasheets import SUBSTRATES  # noqa
OUT = ROOT / "results"
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_count(smi):
    return gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smi)).astype(np.float32)

rows = []
for dname, fname, filt in [("redam", "redam.csv", None), ("electro", "electro.csv", None), ("deamino", "deamino.csv", None),
                           ("UmAS", "UmAS.csv", None), ("UmAS_nature2010", "UmAS.csv", "nature2010_465")]:
    df = pd.read_csv(ROOT / "data/clean" / fname)
    if filt: df = df[df.dataset == filt].reset_index(drop=True)
    base = "UmAS" if dname.startswith("UmAS") else dname
    a, b = SUBSTRATES[base]
    X = np.stack([morgan_count(s1) + morgan_count(s2) for s1, s2 in zip(df[a], df[b])])
    y = df["yield"].to_numpy(float)
    for seed in range(10):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed)
        for mname, model in [("Morgan2048sum-RF (SBR1.5-like feats)", RandomForestRegressor(n_estimators=1000, max_features=1.0, max_depth=10000, min_samples_split=2, n_jobs=-1, random_state=42)),
                             ("Morgan2048sum-ElasticNet (SBR1.5-like)", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=20, cv=5, max_iter=10000, n_jobs=-1, random_state=0))]:
            t = time.time(); model.fit(X[tr], y[tr]); p = model.predict(X[te])
            rows.append(dict(dataset=dname, model=mname, seed=seed, n_train=len(tr), n_test=len(te), MAE=mean_absolute_error(y[te], p),
                             R2=r2_score(y[te], p), RMSE=float(np.sqrt(mean_squared_error(y[te], p))), fit_s=round(time.time() - t, 1)))
            print(f"{dname} seed {seed} {mname}: MAE={rows[-1]['MAE']:.2f} R2={rows[-1]['R2']:.3f} ({rows[-1]['fit_s']}s)", flush=True)
pd.DataFrame(rows).to_csv(OUT / "control_sbr_like_features_metrics_per_seed.csv", index=False)
print("DONE", flush=True)
