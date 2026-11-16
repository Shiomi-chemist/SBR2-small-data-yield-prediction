# Statistical tests for the SBR ChemComm manuscript

Takuma Morimoto, 30 October 2026 (`make_si_tables.py` added 11 November 2026)
Refers to: the Supporting Information of the SBR ChemComm manuscript, Section "Statistical analysis" and Tables S2-S5

## 1. Contents

| File | Content |
|---|---|
| `stats_tests_sbr.py` | The tests. `python stats_tests_sbr.py` regenerates `results/` |
| `make_si_tables.py` | Rebuilds SI Tables S2-S5 from the test results and the per-split accuracies (`results/si_tables.md`, `results/table_S2.csv` ... `table_S5.csv`). Reads the control models from `../mff_benchmark/results/` |
| `check_splits.py` | Confirms that the SBR splits and the MFF splits are identical (the basis for pairing SBR with MFF) |
| `requirements.txt` | Library versions used (numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1) |
| `input/all_splits_all_reactions.csv` | Provided by S. Shiomi. Per-split test MAE and R² of 4 datasets x 6 models (SBR-C2, SBR-2, SBR-2P, SBR-3, Null mean, Null median) x seeds 0-9 |
| `input/outer_test_membership.csv` | Provided by S. Shiomi. Indices of the test set of every split |
| `input/mff_rf_metrics_per_seed.csv` | Per-split test MAE and R² of MFF + random forest (Sandfort et al., *Chem* 2020 settings: 24 fingerprints x 3096 bits, 1000 trees) run on the identical splits |
| `results/paired_wilcoxon_holm.csv` | All tests, long format (one row per comparison); columns below |
| `results/paired_wilcoxon_holm.md` | The same as tables |
| `results/si_tables.md`, `results/table_S*.csv` | SI Tables S2-S5 |

Columns of `paired_wilcoxon_holm.csv`:
`dataset, model_a, model_b, metric (MAE/R2), mean_a, mean_b, mean_diff (= a - b), n_splits_a_better (number of splits in which a was smaller; for R2 this is the number in which a was lower, so read with care), W (Wilcoxon statistic), p_raw (unadjusted), family, p_holm (Holm-adjusted)`.

## 2. Procedure

1. For each dataset and each of the ten splits (seeds 0-9), take the difference between the test MAE of model A and model B on the same split.
2. Apply a two-sided paired Wilcoxon signed-rank test to the ten differences (`scipy.stats.wilcoxon`; exact distribution when no difference is zero, otherwise the approximation).
3. Adjust the p values within each family of comparisons with the Holm procedure.
4. Repeat for R². The main text and the SI report the Holm-adjusted MAE p values; the R² tests are in Table S3 and Table S5.

With ten splits the smallest two-sided exact p value is 2/2^10 = 0.00195, and 15 x 0.00195 = 0.029 is the smallest adjusted p value attainable in a family of 15 comparisons. This is why p = 0.029 appears repeatedly: it means "better in all ten splits; the p value cannot be smaller".

## 3. Families for the Holm correction

| Family | Content | Number of comparisons m |
|---|---|---|
| A (per dataset) | All pairs among the 6 models (SBR-C2, SBR-2, SBR-2P, SBR-3, Null mean, Null median). SBR versus Null and SBR versus SBR are in the same family | 15, in each of 4 datasets |
| B (all datasets) | 4 SBR representations x 4 datasets versus MFF + RF | 16 |

MAE and R² are adjusted separately. Splitting family A into "SBR versus Null" (8) and "SBR versus SBR" (6) would relax the correction, but the DeaminoCpl conclusion would not change (SBR-C2 versus Null mean: unadjusted p = 0.014, adjusted 0.11 even with 8 comparisons). All pairs were kept in one family because they are the comparisons shown together in one panel of Fig. 2 or Fig. 3.

## 4. Validity of paired tests on overlapping splits

The ten splits re-use the same 96-190 reactions, so the ten MAE values are correlated and the p values tend to be smaller than they would be for independent samples (Dietterich 1998, *Neural Comput.*; Nadeau and Bengio 2003, *Mach. Learn.*). Three points:

- A paired test on matched splits is the standard way to compare models in machine learning (Demsar 2006, *J. Mach. Learn. Res.*). It removes the split-to-split variation in difficulty and is fairer than comparing means ± SD across splits.
- Every difference reported as significant is in the same direction in all ten splits (SBR versus Null for RedAm and UmAS; SBR-2P and SBR-3 versus SBR-2 for UmAS) with a large effect (1.5-3.3 MAE points, 23-30 % of the null model), so the conclusions do not depend on the optimism of the p values.
- The differences reported as not significant (DeaminoCpl SBR versus Null, SBR versus MFF, everything for ElectroCpl) did not reach significance even with optimistic p values, so those conclusions are conservative.

The SI therefore states that the p values are descriptive summaries of consistency across splits rather than strict inferential statistics. A corrected resampled t-test (Nadeau and Bengio) or non-overlapping repeated k-fold splits would be stricter alternatives if a reviewer asks; the conclusions are expected to be unchanged.

## 5. Main results (MAE, Holm-adjusted p values)

| Dataset | 4 SBR representations versus Null mean / median | Between representations | Versus MFF + RF |
|---|---|---|---|
| RedAm | all 8 comparisons p = 0.029 (better in 10/10 splits) | SBR-2 < SBR-C2 (p = 0.041); SBR-2 versus SBR-2P / SBR-3 no difference (p > 0.99) | no difference (p >= 0.64) |
| DeaminoCpl | **not significant** (smallest p = 0.21; smallest unadjusted 0.014) | no difference | no difference (smallest p = 0.062; MFF mean MAE 7.44 is below SBR 7.99-8.32) |
| ElectroCpl | no difference (all p > 0.99) | no difference | no difference |
| UmAS | all 8 comparisons p = 0.029 (10/10 splits) | SBR-2P, SBR-3 < SBR-2 (p = 0.029); SBR-2P < SBR-C2 (p = 0.029); SBR-2P versus SBR-3 no difference (p = 0.38) | no difference (p >= 0.36) |

R² gives the same picture, except that for DeaminoCpl the R² of SBR-C2 and SBR-2 is significantly lower than that of MFF + RF (p = 0.031), and the R² of SBR-C2, SBR-2P and SBR-3 is significantly higher than that of the null models; both points are stated in the SI.
