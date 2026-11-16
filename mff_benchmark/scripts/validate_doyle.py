"""Sanity check of the standalone MFF implementation on the original benchmark.

Dataset: Ahneman, Estrada, Lin, Dreher, Doyle, Science 2018 (Buchwald-Hartwig,
3955 reactions). The xlsx shipped with EasyChemML (Examples/_DATASETS) has the
10 random 70/30 splits used by Ahneman et al. and by Sandfort et al. 2020 as
sheets FullCV_01..FullCV_10 (training rows first, then test rows).

Sandfort et al. 2020 (SI Table S13) report per-split R2 of 0.921, 0.920, 0.926, 0.918,
0.932, 0.920, 0.936, 0.926, 0.937, 0.937 (mean 0.927) with RF(5000/10000 trees,
max_features='auto' = all features, 3096-bit fingerprints).

  python scripts/validate_doyle.py --sheets 1-10                 # paper settings, 500 trees (~35 min)
  python scripts/validate_doyle.py --fp-length 2048 --max-features sqrt   # fast variant (~1 min)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mff.encoder import MFFEncoder  # noqa: E402

XLSX = Path(__file__).resolve().parents[1] / "data" / "reference" / "Dreher_and_Doyle_input_data.xlsx"
COMPONENTS = ["Ligand", "Additive", "Base", "Aryl halide"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-length", type=int, default=3096)
    ap.add_argument("--sheets", default="1-10")
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--max-features", default="1.0")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mf = a.max_features
    try:
        mf = float(mf)
    except ValueError:
        pass
    lo, hi = (int(x) for x in a.sheets.split("-")) if "-" in a.sheets else (int(a.sheets),) * 2

    enc = MFFEncoder(fp_length=a.fp_length)
    rows = []
    for k in range(lo, hi + 1):
        df = pd.read_excel(XLSX, sheet_name=f"FullCV_{k:02d}")
        n_train = int(round(a.train_frac * len(df)))
        t0 = time.time()
        X = enc.encode_table([df[c] for c in COMPONENTS]).astype(np.float32)
        keep = X.min(axis=0) != X.max(axis=0)
        X = X[:, keep]
        y = df["Output"].to_numpy(float)
        t_enc = time.time() - t0
        rf = RandomForestRegressor(n_estimators=a.trees, max_features=mf, n_jobs=a.n_jobs, random_state=0)
        t0 = time.time()
        rf.fit(X[:n_train], y[:n_train])
        yp = rf.predict(X[n_train:])
        r2 = r2_score(y[n_train:], yp)
        rmse = np.sqrt(mean_squared_error(y[n_train:], yp))
        rows.append({"sheet": f"FullCV_{k:02d}", "n_train": n_train, "n_test": len(df) - n_train,
                     "n_features": X.shape[1], "r2": r2, "rmse": rmse})
        print(f"FullCV_{k:02d}: n_train={n_train} n_test={len(df)-n_train} features={X.shape[1]} "
              f"R2={r2:.3f} RMSE={rmse:.2f}  (encode {t_enc:.0f}s, fit {time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    print(f"\nmean R2 = {res.r2.mean():.3f} ± {res.r2.std():.3f}, mean RMSE = {res.rmse.mean():.2f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
