"""Paired statistical tests for the SBR ChemComm manuscript.

Input : input/all_splits_all_reactions.csv   (S. Shiomi, per-split MAE/R2 of SBR-C2/-2/-2P/-3 and the two null models, 70:30 splits seed 0-9)
        input/mff_rf_metrics_per_seed.csv    (MFF + random forest, Sandfort et al. 2020 settings, on the identical splits; see check_splits.py)
Tests : two-sided paired Wilcoxon signed-rank test across the ten splits (one pair per seed),
        Holm correction within each family of comparisons.
Families:
  A  per dataset, all 15 pairwise comparisons among the 6 models (SBR-C2, SBR-2, SBR-2P, SBR-3, Null mean, Null median)
  B  SBR vs MFF+RF: 4 representations x 4 datasets = 16 comparisons
Output: results/paired_wilcoxon_holm.csv (all tests, long format) and results/paired_wilcoxon_holm.md (tables).
Run   : python stats_tests_sbr.py      (needs numpy, pandas, scipy; see requirements.txt)
"""
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"; OUT.mkdir(parents=True, exist_ok=True)
sbr = pd.read_csv(ROOT / "input/all_splits_all_reactions.csv")
mff = pd.read_csv(ROOT / "input/mff_rf_metrics_per_seed.csv")
DSMAP = {"redam": "RedAm", "deamino": "DeaminoCpl", "electro": "ElectroCpl", "UmAS": "UmAS"}
mff = mff.assign(dataset=lambda d: d.dataset.map(DSMAP), model="MFF+RF").dropna(subset=["dataset"])
allm = pd.concat([sbr[["dataset", "seed", "model", "MAE", "R2"]], mff[["dataset", "seed", "model", "MAE", "R2"]]])
MODELS = ["SBR-C2", "SBR-2", "SBR-2P", "SBR-3", "Null mean", "Null median"]
DATASETS = ["RedAm", "DeaminoCpl", "ElectroCpl", "UmAS"]

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj

def paired(ds, a, b, metric="MAE"):
    d = allm[allm.dataset == ds].pivot(index="seed", columns="model", values=metric)
    x, y = d[a].to_numpy(), d[b].to_numpy()
    diff = x - y
    res = wilcoxon(x, y, alternative="two-sided", method="exact" if np.all(diff != 0) else "auto")
    return dict(dataset=ds, model_a=a, model_b=b, metric=metric, mean_a=x.mean(), mean_b=y.mean(),
                mean_diff=diff.mean(), n_splits_a_better=int((diff < 0).sum()), W=float(res.statistic), p_raw=float(res.pvalue))

rows = []
for metric in ("MAE", "R2"):
    for ds in DATASETS:
        fam = [paired(ds, a, b, metric) for a, b in combinations(MODELS, 2)]
        padj = holm([r["p_raw"] for r in fam])
        for r, q in zip(fam, padj): r.update(family=f"A_pairwise6_{ds}", p_holm=q)
        rows += fam
    famB = [paired(ds, rep, "MFF+RF", metric) for ds in DATASETS for rep in MODELS[:4]]
    padj = holm([r["p_raw"] for r in famB])
    for r, q in zip(famB, padj): r.update(family="B_SBR_vs_MFF", p_holm=q)
    rows += famB
res = pd.DataFrame(rows)
res.to_csv(OUT / "paired_wilcoxon_holm.csv", index=False)

def fmt(p): return "< 0.001" if p < 0.001 else f"= {p:.3f}"
lines = ["# Paired Wilcoxon signed-rank tests (two-sided, n = 10 splits), Holm-corrected\n"]
for metric in ("MAE", "R2"):
    lines.append(f"\n## {metric}\n")
    for ds in DATASETS:
        lines.append(f"\n### {ds} (family A: 15 pairwise comparisons)\n")
        lines.append("| A | B | mean A | mean B | diff | splits A<B | p raw | p Holm |\n|--|--|--|--|--|--|--|--|")
        for _, r in res[(res.metric == metric) & (res.family == f"A_pairwise6_{ds}")].iterrows():
            lines.append(f"| {r.model_a} | {r.model_b} | {r.mean_a:.2f} | {r.mean_b:.2f} | {r.mean_diff:+.2f} | {r.n_splits_a_better}/10 | {r.p_raw:.4f} | {r.p_holm:.4f} |")
    lines.append(f"\n### SBR vs MFF+RF (family B: 16 comparisons)\n")
    lines.append("| dataset | SBR | mean SBR | mean MFF | diff | splits SBR<MFF | p raw | p Holm |\n|--|--|--|--|--|--|--|--|")
    for _, r in res[(res.metric == metric) & (res.family == "B_SBR_vs_MFF")].iterrows():
        lines.append(f"| {r.dataset} | {r.model_a} | {r.mean_a:.2f} | {r.mean_b:.2f} | {r.mean_diff:+.2f} | {r.n_splits_a_better}/10 | {r.p_raw:.4f} | {r.p_holm:.4f} |")
(OUT / "paired_wilcoxon_holm.md").write_text("\n".join(lines))
print("\n".join(lines))
