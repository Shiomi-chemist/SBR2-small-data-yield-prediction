"""MFF (Sandfort et al. 2020) on the four SBR datasets, with the SBR evaluation protocol.

Protocol (Shiomi et al. ChemRxiv 2026, 10.26434/chemrxiv.15000193/v1, and S. Shiomi's code):
  * sklearn.model_selection.train_test_split(np.arange(n), test_size=0.30, random_state=seed), seeds 0-9
    (verified: Null-mean / Null-median MAEs reproduce S. Shiomi's numbers to 3 decimals)
  * metrics on the 30 % test set: MAE (primary), R2; RMSE added
Variants evaluated on exactly those splits:
  null_mean, null_median           train-set mean / median predicted for every test reaction
  MFF-RF (all components)          original MFF: every SMILES column (substrates + reagents + solvent), RF paper settings
  MFF-RF (substrates)              only the two substrate columns (same information as SBR)
  MFF-RF (all + conditions)        all components + temperature_C, time_h appended
  MFF-ElasticNet (substrates)      MFF bits of the substrates + ElasticNetCV (5-fold) = SBR's regressor on MFF features
RF settings follow the paper (max_features = all, max_depth = 10000, min_samples_split = 2); n_estimators = 1000
(the paper picks 5000 / 10000 by inner CV; at n_train ~ 70-130 the extra trees change nothing but run time).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mff.encoder import MFFEncoder  # noqa: E402
from scripts.clean_shiomi_datasheets import SUBSTRATES  # noqa: E402

OUT = ROOT / "results"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(10); TEST_SIZE = 0.30; FP_LENGTH = 3096; N_TREES = 1000; N_JOBS = -1
DATASETS = {
    "redam": "redam.csv", "electro": "electro.csv", "deamino": "deamino.csv",
    "UmAS": "UmAS.csv",                       # all 190 literature reactions (8 reports)
    "UmAS_nature2010": "UmAS.csv",            # Johnston 2010 only (the preprint's single-report case)
}
COND = ["temperature_C", "time_h"]


def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))))


def rf():
    return RandomForestRegressor(n_estimators=N_TREES, max_features=1.0, max_depth=10000, min_samples_split=2,
                                 n_jobs=N_JOBS, random_state=42)


def enet():
    return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=20, cv=5, max_iter=10000,
                        n_jobs=N_JOBS, random_state=0)


def drop_const(X):
    return X[:, X.min(axis=0) != X.max(axis=0)]


enc = MFFEncoder(FP_LENGTH)
rows, preds, splits = [], [], []
for dname, fname in DATASETS.items():
    df = pd.read_csv(ROOT / "data/clean" / fname)
    if dname == "UmAS_nature2010":
        df = df[df["dataset"] == "nature2010_465"].reset_index(drop=True)
    base = "UmAS" if dname.startswith("UmAS") else dname
    y = df["yield"].to_numpy(float)
    smi_all = [c for c in df.columns if "smiles" in c]
    smi_sub = SUBSTRATES[base]
    t0 = time.time()
    X_all = drop_const(enc.encode_table([df[c] for c in smi_all]).astype(np.float32))
    X_sub = drop_const(enc.encode_table([df[c] for c in smi_sub]).astype(np.float32))
    X_cond = np.hstack([X_all, df[COND].fillna(0).to_numpy(np.float32)])
    print(f"[{dname}] n={len(df)} | MFF all: {len(smi_all)} cols -> {X_all.shape[1]} informative bits | "
          f"substrates -> {X_sub.shape[1]} bits | encode {time.time()-t0:.1f}s", flush=True)
    feats = {"MFF-RF (all components)": (X_all, rf), "MFF-RF (substrates)": (X_sub, rf),
             "MFF-RF (all + conditions)": (X_cond, rf), "MFF-ElasticNet (substrates)": (X_sub, enet)}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=seed)
        splits += [dict(dataset=dname, seed=seed, row_index=int(i), reaction_id=df.reaction_id[i], set=s)
                   for idx, s in ((tr, "train"), (te, "test")) for i in idx]
        for mname, p in (("Null mean", np.full(len(te), y[tr].mean())), ("Null median", np.full(len(te), np.median(y[tr])))):
            rows.append(dict(dataset=dname, model=mname, seed=seed, n_train=len(tr), n_test=len(te), **metrics(y[te], p)))
        for mname, (X, make) in feats.items():
            t1 = time.time(); m = make(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            rec = dict(dataset=dname, model=mname, seed=seed, n_train=len(tr), n_test=len(te), **metrics(y[te], p),
                       fit_s=round(time.time() - t1, 1))
            if hasattr(m, "alpha_"): rec.update(enet_alpha=float(m.alpha_), enet_l1_ratio=float(m.l1_ratio_))
            rows.append(rec)
            preds.append(pd.DataFrame(dict(dataset=dname, model=mname, seed=seed, row_index=te,
                                           reaction_id=df.reaction_id.to_numpy()[te], y_true=y[te], y_pred=p)))
            print(f"   seed {seed} {mname:28s} MAE={rec['MAE']:.2f} R2={rec['R2']:.3f} ({rec['fit_s']}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "mff_metrics_per_seed.csv", index=False)   # checkpoint
pd.DataFrame(rows).to_csv(OUT / "mff_metrics_per_seed.csv", index=False)
pd.concat(preds).to_csv(OUT / "mff_predictions_test.csv", index=False)
pd.DataFrame(splits).to_csv(OUT / "splits_70_30_seed0-9.csv", index=False)
json.dump(dict(test_size=TEST_SIZE, seeds=list(SEEDS), fp_length=FP_LENGTH, n_trees=N_TREES,
               rf="max_features=1.0,max_depth=10000,min_samples_split=2,random_state=42"), open(OUT / "run_config.json", "w"), indent=2)
print("DONE", flush=True)
