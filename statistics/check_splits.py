"""Verify that the test-set membership used for the SBR results (input/outer_test_membership.csv)
is identical to sklearn's train_test_split(np.arange(n), test_size=0.30, random_state=seed), seed 0-9,
which is the split used for the MFF + random forest runs. This is what makes the SBR-vs-MFF tests paired."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent
m = pd.read_csv(ROOT / "input/outer_test_membership.csv")
N = {"RedAm": 118, "DeaminoCpl": 96, "ElectroCpl": 97, "UmAS": 190}
ok = True
for ds, n in N.items():
    for seed in range(10):
        te = set(train_test_split(np.arange(n), test_size=0.30, random_state=seed)[1])
        got = set(m[(m.dataset == ds) & (m.seed == seed)]["num"])
        if te != got:
            ok = False; print("MISMATCH", ds, seed)
print("all 40 test sets identical:", ok)
