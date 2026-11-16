"""Rebuild SI Tables S2-S5 (Supporting Information, sections "Statistical analysis" and "Comparison with MFF") from the per-split results.

Input : input/all_splits_all_reactions.csv          per-split MAE / R2 of SBR-C2, SBR-2, SBR-2P, SBR-3, Null mean, Null median (S. Shiomi)
        input/mff_rf_metrics_per_seed.csv           MFF + random forest on the same splits
        ../mff_benchmark/results/mff_metrics_per_seed.csv                    MFF + ElasticNet (control)
        ../mff_benchmark/results/control_sbr_like_features_metrics_per_seed.csv   SBR-2 features + random forest (control)
        results/paired_wilcoxon_holm.csv            output of stats_tests_sbr.py
Output: results/si_tables.md  (Tables S2-S5 as markdown)  and results/table_S2.csv ... table_S5.csv
Run   : python stats_tests_sbr.py && python make_si_tables.py
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
MFF_DIR = ROOT.parent / "mff_benchmark" / "results"
DS = ["RedAm", "DeaminoCpl", "ElectroCpl", "UmAS"]
DSMAP = {"redam": "RedAm", "deamino": "DeaminoCpl", "electro": "ElectroCpl", "UmAS": "UmAS"}
MODELS = ["SBR-C2", "SBR-2", "SBR-2P", "SBR-3", "Null mean", "Null median"]
PAIRS = [(a, b) for a in MODELS[:4] for b in MODELS[4:]] + [("SBR-C2", "SBR-2"), ("SBR-C2", "SBR-2P"), ("SBR-C2", "SBR-3"),
                                                            ("SBR-2", "SBR-2P"), ("SBR-2", "SBR-3"), ("SBR-2P", "SBR-3"), ("Null mean", "Null median")]

stats = pd.read_csv(OUT / "paired_wilcoxon_holm.csv")
sbr = pd.read_csv(ROOT / "input/all_splits_all_reactions.csv")[["dataset", "seed", "model", "MAE", "R2"]]
mff = pd.read_csv(ROOT / "input/mff_rf_metrics_per_seed.csv").assign(model="MFF (random forest)")
extra = []
mm = pd.read_csv(MFF_DIR / "mff_metrics_per_seed.csv")
extra.append(mm[mm.model == "MFF-ElasticNet (substrates)"].assign(model="MFF + ElasticNet"))
cc = pd.read_csv(MFF_DIR / "control_sbr_like_features_metrics_per_seed.csv")
extra.append(cc[cc.model == "Morgan2048sum-RF (SBR1.5-like feats)"].assign(model="SBR-2 features + random forest"))
ours = pd.concat([mff] + extra)[["dataset", "seed", "model", "MAE", "R2"]]
ours["dataset"] = ours.dataset.map(DSMAP); ours = ours.dropna(subset=["dataset"])
allm = pd.concat([sbr, ours])


def fp(p): return "> 0.99" if p >= 0.995 else f"{p:.3f}"
def fd(x): return f"{x:+.2f}"
def star(p): return f"**{fp(p)}**" if p < 0.05 else fp(p)
def stat(ds, a, b, metric, fam):
    r = stats[(stats.dataset == ds) & (stats.model_a == a) & (stats.model_b == b) & (stats.metric == metric) & stats.family.str.startswith(fam)]
    assert len(r) == 1, (ds, a, b, metric, fam); return r.iloc[0]


md = ["# SI Tables S2-S5 rebuilt from the per-split results\n",
      "p = Holm-adjusted two-sided paired Wilcoxon signed-rank p value (n = 10 splits); bold = p < 0.05. "
      "Δ = mean over splits of (A − B). Mean ± SD over the ten splits.\n"]

# ---- Tables S2 (MAE) and S3 (R2): 15 pairwise comparisons per dataset ----------------------------------
for metric, lab, tno in (("MAE", "MAE", 2), ("R2", "R²", 3)):
    rows = []
    for a, b in PAIRS:
        rec = {"comparison": f"{a} vs {b}"}
        for d in DS:
            r = stat(d, a, b, metric, "A_")
            rec[f"{d} d{lab}"] = round(r.mean_diff, 3); rec[f"{d} p"] = round(r.p_holm, 4)
        rows.append(rec)
    t = pd.DataFrame(rows); t.to_csv(OUT / f"table_S{tno}.csv", index=False)
    md.append(f"\n## Table S{tno}. Pairwise comparisons of test {lab} (family A, Holm over 15 comparisons per dataset)\n")
    md.append("| Comparison (A vs B) | " + " | ".join(f"{d} Δ{lab} | {d} p" for d in DS) + " |")
    md.append("|---|" + "---|" * (2 * len(DS)))
    for a, b in PAIRS:
        cells = []
        for d in DS:
            r = stat(d, a, b, metric, "A_"); cells += [fd(r.mean_diff), star(r.p_holm)]
        md.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")

# ---- Table S4: accuracy of all models ------------------------------------------------------------------
spec = ["SBR-C2", "SBR-2", "SBR-2P", "SBR-3", "MFF (random forest)", "MFF + ElasticNet", "SBR-2 features + random forest", "Null mean", "Null median"]
g = allm.groupby(["dataset", "model"]).agg(mean_MAE=("MAE", "mean"), sd_MAE=("MAE", lambda x: x.std(ddof=1)),
                                           mean_R2=("R2", "mean"), sd_R2=("R2", lambda x: x.std(ddof=1)), n=("seed", "count"))
assert (g.n == 10).all()
rows = []
for m in spec:
    for d in DS:
        r = g.loc[(d, m)]
        rows.append(dict(model=m, dataset=d, mean_MAE=round(r.mean_MAE, 3), sd_MAE=round(r.sd_MAE, 3), mean_R2=round(r.mean_R2, 3), sd_R2=round(r.sd_R2, 3)))
pd.DataFrame(rows).to_csv(OUT / "table_S4.csv", index=False)
md.append("\n## Table S4. Test MAE (percentage points) and R² of all models, mean ± SD over the ten splits\n")
for metric, lab in (("MAE", "MAE"), ("R2", "R²")):
    md.append(f"\n**Test {lab}**\n\n| Model | " + " | ".join(DS) + " |\n|---|" + "---|" * len(DS))
    for m in spec:
        md.append(f"| {m} | " + " | ".join(f"{g.loc[(d, m)][f'mean_{metric}']:.2f} ± {g.loc[(d, m)][f'sd_{metric}']:.2f}" for d in DS) + " |")

# ---- Table S5: SBR representations vs MFF ---------------------------------------------------------------
rows = []
md.append("\n## Table S5. SBR representations versus MFF (random forest) (family B, Holm over 16 comparisons per metric)\n")
for metric, lab in (("MAE", "MAE"), ("R2", "R²")):
    md.append(f"\n**Test {lab}**\n\n| SBR representation | " + " | ".join(f"{d} Δ{lab} | {d} p" for d in DS) + " |\n|---|" + "---|" * (2 * len(DS)))
    for a in MODELS[:4]:
        cells = []
        for d in DS:
            r = stat(d, a, "MFF+RF", metric, "B_"); cells += [fd(r.mean_diff), star(r.p_holm)]
            rows.append(dict(metric=metric, representation=a, dataset=d, diff=round(r.mean_diff, 3), p_holm=round(r.p_holm, 4)))
        md.append(f"| {a} | " + " | ".join(cells) + " |")
pd.DataFrame(rows).to_csv(OUT / "table_S5.csv", index=False)

(OUT / "si_tables.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
