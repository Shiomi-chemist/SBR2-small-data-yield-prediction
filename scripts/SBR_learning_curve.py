"""Code S3. Nested-subset learning curves for SBR-2 and two null models.
 
Ten outer 80:20 splits, each with ten random orderings of the training
pool. Within each ordering, training subsets of increasing size are
nested, and ElasticNet is reoptimized via five-fold CV for EVERY
subset. MAEs are averaged over the ten repetitions within each outer
split; final mean +/- SD are calculated across the ten outer splits.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, ParameterGrid, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
#DATASET = "ML_redam"
#DATASET = "ML_deamino"
#DATASET = "ML_electro"
DATASET = "ML_UmAS"

INPUT_DIR = Path("SBR_features")
OUTPUT_DIR = Path("SBR_learning_curve_results")
MODEL = "SBR-2"

#test_analysis
#OUTER_SEEDS = [0, 1]
#INNER_SEEDS = [0, 1]
#TRAIN_SIZES = [20, 40]

# Full analysis settings (Fig. 4)
OUTER_SEEDS = list(range(10))
INNER_SEEDS = list(range(10))
TRAIN_SIZES = list(range(20, 191, 10)) + ["max"]

OUTER_TEST_SIZE = 0.20
N_FOLDS = 5
N_JOBS = -1
ALPHAS = np.arange(0.1, 1.0, 0.1)
L1_RATIOS = np.arange(0.1, 1.0, 0.1)
PARAMETERS = list(ParameterGrid({"alpha": ALPHAS, "l1_ratio": L1_RATIOS}))
CV = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
 
 
def pipeline(alpha, l1_ratio):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=500000, tol=1e-3,
            selection="random", random_state=0,
        )),
    ])
 
 
def fit_optimized(X_train, y_train):
    best_score, best_params = -np.inf, None
    for params in PARAMETERS:
        scores = cross_val_score(
            pipeline(**params), X_train, y_train,
            cv=CV, scoring="neg_mean_absolute_error", n_jobs=N_JOBS,
        )
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_params = score, params
    model = pipeline(**best_params)
    model.fit(X_train, y_train)
    return model, best_params
 
 
def main():
    path = INPUT_DIR / f"{DATASET}_{MODEL}.csv"
    df = pd.read_csv(path)
    if df["num"].isna().any() or df["num"].duplicated().any():
        raise ValueError("'num' must be unique and nonmissing")
    cols = [c for c in df.columns if c.startswith("Morgan_")]
    if len(cols) != 2048:
        raise ValueError(f"Expected 2048 fingerprint features; found {len(cols)}")
    X = df[cols].to_numpy(dtype=float)
    y = pd.to_numeric(df["yield"], errors="coerce").to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Input contains missing/non-finite features or yields")
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for outer_seed in OUTER_SEEDS:
        pool_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=OUTER_TEST_SIZE,
            random_state=outer_seed, shuffle=True,
        )
        max_n = len(pool_idx)
        sizes = sorted({max_n if size == "max" else size
                        for size in TRAIN_SIZES
                        if size == "max" or size <= max_n})
        y_test = y[test_idx]
        for inner_seed in INNER_SEEDS:
            # Each random ordering gives nested subsets across sizes.
            rng = np.random.default_rng(1000003 * outer_seed + inner_seed)
            ordered = rng.permutation(pool_idx)
            for train_n in sizes:
                train_idx = ordered[:train_n]
                y_train = y[train_idx]
                nulls = (("Null mean", y_train.mean()),
                         ("Null median", np.median(y_train)))
                for label, constant in nulls:
                    pred = np.full(y_test.shape, constant, dtype=float)
                    records.append(dict(outer_seed=outer_seed,
                                        inner_seed=inner_seed, train_n=train_n,
                                        model=label,
                                        MAE=mean_absolute_error(y_test, pred)))
                fitted, params = fit_optimized(X[train_idx], y_train)
                pred = fitted.predict(X[test_idx])
                records.append(dict(outer_seed=outer_seed,
                                    inner_seed=inner_seed, train_n=train_n,
                                    model=MODEL,
                                    MAE=mean_absolute_error(y_test, pred),
                                    alpha=params["alpha"],
                                    l1_ratio=params["l1_ratio"]))
        print(f"Completed outer split {outer_seed}")
 
    evaluations = pd.DataFrame(records)
    # Average first across inner draws, then calculate mean and SD across outer splits.
    outer_mean = (evaluations.groupby(["model", "train_n", "outer_seed"],
                                       as_index=False)
                  .agg(MAE=("MAE", "mean")))
    summary = (outer_mean.groupby(["model", "train_n"], as_index=False)
               .agg(mean_MAE=("MAE", "mean"),
                    std_outer_MAE=("MAE", "std")))
 
    evaluations.to_csv(OUTPUT_DIR / f"{DATASET}_all_evaluations.csv", index=False)
    outer_mean.to_csv(OUTPUT_DIR / f"{DATASET}_outer_means.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"{DATASET}_learning_curve_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
 
 
if __name__ == "__main__":
    main()

