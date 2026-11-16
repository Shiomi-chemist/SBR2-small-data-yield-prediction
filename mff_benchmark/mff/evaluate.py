"""Evaluate MFF yield prediction on a reaction data sheet.

Usage (from the repo root, venv activated):

  python -m mff.evaluate --input data/example_datasheet.csv \
      --components aryl_halide_smiles amine_smiles ligand_smiles base_smiles solvent_smiles \
      --target yield --models rf --seeds 0-9 --test-size 0.3 --out results/example

Input: CSV or XLSX (use --sheet for XLSX). One row per reaction, one column per
reaction component containing a SMILES string (empty cell = component absent),
and one numeric target column (yield). Optional --numeric-cols are appended
as-is to the MFF features (not part of the original MFF, off by default).

Splits: sklearn train_test_split(test_size, random_state=seed, shuffle=True)
for every seed, OR --split-file (CSV with columns: seed, row_index, set) to
reuse exactly the splits of another method (e.g. SBR).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .encoder import MFFEncoder


# ----------------------------------------------------------------------------
def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_table(path: str, sheet: str | None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet or 0)
    return pd.read_csv(p)


def build_model(name: str, n_jobs: int, seed: int, rf_trees: int, rf_max_features, enet_alpha, enet_l1):
    name = name.lower()
    if name == "elasticnet":
        if enet_alpha is None:
            # inner 5-fold CV on the training set for alpha and l1_ratio
            return ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], alphas=50,
                                cv=5, max_iter=20000, n_jobs=n_jobs, random_state=seed)
        return ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1, max_iter=20000, random_state=seed)
    if name == "rf":
        # Sandfort et al. 2020 (SI): max_features='auto' (= all), max_depth=10000,
        # min_samples_split=2, criterion='mse', n_estimators in {5000, 10000} by inner CV
        return RandomForestRegressor(n_estimators=rf_trees, max_features=rf_max_features,
                                     max_depth=10000, min_samples_split=2,
                                     n_jobs=n_jobs, random_state=seed)
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=rf_trees, max_features=rf_max_features,
                                   n_jobs=n_jobs, random_state=seed)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, loss_function="RMSE",
                                 verbose=False, random_seed=seed, thread_count=n_jobs,
                                 allow_writing_files=False)
    raise ValueError(f"unknown model {name}")


def metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default=None, help="sheet name for xlsx input")
    ap.add_argument("--components", nargs="+", required=True, help="SMILES columns (reaction components)")
    ap.add_argument("--target", required=True, help="numeric target column (e.g. yield)")
    ap.add_argument("--id-col", default=None, help="optional reaction id column kept in prediction output")
    ap.add_argument("--numeric-cols", nargs="*", default=[], help="extra numeric columns appended to MFF")
    ap.add_argument("--fp-length", type=int, default=3096, help="bits per fingerprint (paper: 3096)")
    ap.add_argument("--models", nargs="+", default=["rf"],
                    choices=["elasticnet", "rf", "extratrees", "catboost"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--split-file", default=None, help="CSV: seed,row_index,set(train|test)")
    ap.add_argument("--rf-trees", type=int, default=500,
                    help="paper: 5000 or 10000 chosen by 5-fold CV; 500 is ~10x faster and within noise")
    ap.add_argument("--rf-max-features", default="1.0",
                    help="sklearn max_features: 1.0 = all features (paper: 'auto'), 'sqrt', 'log2', or a fraction")
    ap.add_argument("--enet-alpha", type=float, default=None, help="fixed alpha (default: ElasticNetCV)")
    ap.add_argument("--enet-l1-ratio", type=float, default=0.5)
    ap.add_argument("--keep-constant-columns", action="store_true",
                    help="keep constant MFF bits (default drops them; label-free, and exactly equivalent "
                         "for RF with max_features=1.0 since constant bits are never chosen as splits)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    mf = args.rf_max_features
    try:
        mf = float(mf)
        if mf == 1.0:
            mf = 1.0
    except ValueError:
        pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(args.input, args.sheet)
    missing = [c for c in args.components + [args.target] + args.numeric_cols if c not in df.columns]
    if missing:
        sys.exit(f"columns not found in input: {missing}\navailable: {list(df.columns)}")
    df = df.reset_index(drop=True)
    y = pd.to_numeric(df[args.target], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y).any():
        bad = df.index[np.isnan(y)].tolist()
        sys.exit(f"non-numeric / empty target in rows (0-based): {bad[:20]}")

    print(f"[MFF] {len(df)} reactions, components={args.components}, fp_length={args.fp_length}")
    t0 = time.time()
    enc = MFFEncoder(fp_length=args.fp_length)
    X = enc.encode_table([df[c] for c in args.components]).astype(np.float32)
    print(f"[MFF] encoded: X shape {X.shape} in {time.time() - t0:.1f}s")
    if not args.keep_constant_columns:
        keep = X.min(axis=0) != X.max(axis=0)
        X = X[:, keep]
        print(f"[MFF] dropped constant bits -> {X.shape[1]} informative features")
    if args.numeric_cols:
        Z = df[args.numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        X = np.hstack([X, Z])
        print(f"[MFF] appended numeric columns {args.numeric_cols} -> {X.shape[1]} features")

    seeds = parse_seeds(args.seeds)
    split_df = pd.read_csv(args.split_file) if args.split_file else None

    rows, preds = [], []
    for model_name in args.models:
        for seed in seeds:
            if split_df is not None:
                s = split_df[split_df["seed"] == seed]
                tr = s.loc[s["set"].str.lower() == "train", "row_index"].to_numpy(int)
                te = s.loc[s["set"].str.lower() == "test", "row_index"].to_numpy(int)
            else:
                tr, te = train_test_split(np.arange(len(df)), test_size=args.test_size,
                                          random_state=seed, shuffle=True)
            model = build_model(model_name, args.n_jobs, seed, args.rf_trees, mf,
                                args.enet_alpha, args.enet_l1_ratio)
            t1 = time.time()
            model.fit(X[tr], y[tr])
            yp_te = model.predict(X[te])
            yp_tr = model.predict(X[tr])
            m_te, m_tr = metrics(y[te], yp_te), metrics(y[tr], yp_tr)
            rec = {"model": model_name, "seed": seed, "n_train": len(tr), "n_test": len(te),
                   **{f"test_{k}": v for k, v in m_te.items()},
                   **{f"train_{k}": v for k, v in m_tr.items()},
                   "fit_seconds": round(time.time() - t1, 1)}
            if model_name == "elasticnet" and hasattr(model, "alpha_"):
                rec["enet_alpha"] = float(model.alpha_)
                rec["enet_l1_ratio"] = float(model.l1_ratio_)
            rows.append(rec)
            print(f"  {model_name:10s} seed={seed}  test R2={m_te['r2']:.3f} RMSE={m_te['rmse']:.2f} "
                  f"MAE={m_te['mae']:.2f}  ({rec['fit_seconds']}s)")
            p = pd.DataFrame({"model": model_name, "seed": seed, "row_index": te,
                              "y_true": y[te], "y_pred": yp_te})
            if args.id_col and args.id_col in df.columns:
                p.insert(2, args.id_col, df.loc[te, args.id_col].to_numpy())
            preds.append(p)

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_per_seed.csv", index=False)
    summary = res.groupby("model")[["test_r2", "test_rmse", "test_mae"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "metrics_summary.csv")
    pd.concat(preds).to_csv(out / "predictions_test.csv", index=False)
    with open(out / "run_config.json", "w") as f:
        json.dump({**vars(args), "n_features": int(X.shape[1]), "n_reactions": int(len(df))}, f, indent=2)
    print("\n=== summary (test set, mean ± SD over seeds) ===")
    print(summary.round(3).to_string())
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
