# SBR ChemComm manuscript: statistical tests and MFF benchmark

Takuma Morimoto, 11 November 2026
Refers to: the SBR ChemComm manuscript and its Supporting Information (SI Section "Statistical analysis", Section "Comparison with MFF", Tables S2-S5)

This repository contains the code, input data, and results supporting the statistical analyses and MFF benchmark described in the Supporting Information. The corresponding results and Tables S2–S5 can be regenerated using the provided scripts.

| Folder | Content | SI |
|---|---|---|
| `statistics/` | Paired Wilcoxon signed-rank tests with Holm correction, and a script that rebuilds SI Tables S2-S5 | "Statistical analysis"; Tables S2, S3, S5 (and the layout of S4) |
| `mff_benchmark/` | Re-implementation of MFF (Sandfort et al., *Chem* 2020), evaluation on the SBR splits, cleaned data, per-split results | "Comparison with MFF"; Table S4 |

## 1. statistics/

| File | Content |
|---|---|
| `README.md` | Description of the tests (procedure, definition of the Holm families, notes on the validity of paired tests on overlapping splits) |
| `stats_tests_sbr.py` | The tests. `python stats_tests_sbr.py` regenerates `results/paired_wilcoxon_holm.{csv,md}` |
| `make_si_tables.py` | Builds **SI Tables S2-S5** from the test results and the per-split accuracies. Writes `results/si_tables.md` and `results/table_S2.csv` ... `table_S5.csv`; the SI tables were transcribed from these |
| `check_splits.py` | Verifies that the SBR test sets (`input/outer_test_membership.csv`) are identical to the sklearn `train_test_split` sets (seeds 0-9) used for MFF, for all 40 dataset-split combinations |
| `input/all_splits_all_reactions.csv` | Provided by S. Shiomi. Per-split test MAE and R² of SBR-C2, SBR-2, SBR-2P, SBR-3, Null mean and Null median |
| `input/outer_test_membership.csv` | Provided by S. Shiomi. Test-set membership of every split |
| `input/mff_rf_metrics_per_seed.csv` | Per-split test MAE and R² of MFF + random forest (the "MFF-RF (all components)" rows of `mff_benchmark/results/mff_metrics_per_seed.csv`) |
| `results/` | Output of the scripts. `paired_wilcoxon_holm.csv` lists every test (MAE and R², raw and Holm-adjusted p values); `si_tables.md` contains Tables S2-S5 |


Run the statistical analysis from the `statistics/` directory:

```bash
cd statistics
python stats_tests_sbr.py
python make_si_tables.py
```

The second script reads the control-model results from `../mff_benchmark/results/`. Keep the original directory structure unchanged.


## 2. mff_benchmark/

### Summary of the method (SI section "Comparison with MFF")

- **Features.** MFF of Sandfort et al.: 24 RDKit fingerprints per molecule (RDKit topological, branched and linear, path lengths 2/4/6/8; Morgan with and without feature invariants, radii 0/2/4/6; layered, path lengths 2/4/6/8; Avalon; MACCS; hashed atom pairs; hashed topological torsions), each with 3096 bits (MACCS: 167 bits, of which bit 0 is always zero), giving the 71,374 bits per molecule of the original work. The fingerprints of all reaction components included in the datasets (substrates, reagents, catalysts, and solvents) were concatenated, and features that were constant across each dataset were removed. As reaction components varied in the DeaminoCpl and UmAS datasets, the resulting MFF representations could contain information beyond the substrate structures used in SBR.
- **Encoder validation.** `mff/encoder.py` was checked to be bit-identical to the `FingerprintGenerator` of EasyChemML, the authors' public code (`scripts/check_equivalence_easychemml.py`; clone EasyChemML and run it). On the Buchwald-Hartwig data of Ahneman et al. (`data/reference/`) it reproduces the R² reported in the original SI (split 1: 0.920 here versus 0.921; `scripts/validate_doyle.py`).
- **Model.** `RandomForestRegressor(max_features=all features, max_depth=10000, min_samples_split=2)` as in the original work, 1000 trees, `random_state=42`. The original work chose 5000 or 10,000 trees by inner cross-validation; at n = 70-130 the extra trees change only the run time, so the number was fixed.
- **Splits.** Identical to SBR: `train_test_split(np.arange(n), test_size=0.30, random_state=seed)`, seeds 0-9, rows in data-sheet order.
- **Control models** (to separate the effect of the representation from that of the regressor): MFF features + ElasticNetCV (5-fold, l1_ratio in {0.1, 0.5, 0.9}, 20 values of alpha), and the SBR-2 features (Morgan count fingerprint, radius 2, 2048 bits, summed over the two substrates) + the same random forest.
- **Environment.** Python 3.13.0, RDKit 2026.09.1, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1 (macOS, arm64).

### Files

| File | Content |
|---|---|
| `mff/encoder.py` | MFF encoder (the 24 fingerprint definitions follow `MFF.py` of EasyChemML) |
| `mff/evaluate.py` | General command-line tool (MFF + RF on any data sheet). The results in this archive were produced with `run_sbr_comparison.py` below |
| `scripts/clean_shiomi_datasheets.py` | `data/raw/MFF_*.xlsx` (the four sheets provided by S. Shiomi) to `data/clean/*.csv`: drops empty rows and columns, corrects inconsistent ring-closure numbering in the UmAS THF/water solvent SMILES (e.g., `C1CCCO2.[H]O[H]` to `C1CCCO1.[H]O[H]`), checks every SMILES with RDKit |
| `scripts/run_sbr_comparison.py` | **Main script.** For each of the four datasets and seeds 0-9: Null mean, Null median, MFF-RF (all components), MFF-RF (substrates), MFF-RF (all components + temperature and time), MFF-ElasticNet (substrates). Writes `results/mff_metrics_per_seed.csv`, `mff_predictions_test.csv`, `splits_70_30_seed0-9.csv`, `run_config.json` |
| `scripts/run_sbr_like_features_rf.py` | Control: SBR-2 features + RF and + ElasticNet. Writes `results/control_sbr_like_features_metrics_per_seed.csv` |
| `scripts/validate_doyle.py` | Reproduction check on the Buchwald-Hartwig data |
| `scripts/check_equivalence_easychemml.py` | Bit-level check against EasyChemML |
| `data/raw/`, `data/clean/` | Data sheets provided by S. Shiomi and the cleaned CSV files |
| `data/reference/Dreher_and_Doyle_input_data.xlsx` | Buchwald-Hartwig data shipped with EasyChemML (validation only) |
| `results/mff_metrics_per_seed.csv` | Per-split MAE, R² and RMSE of all models. The `UmAS_nature2010` rows are a trial restricted to the 42 reactions of Johnston 2010 and are not used in the SI |
| `results/control_sbr_like_features_metrics_per_seed.csv` | Per-split values of the control models |
| `results/mff_predictions_test.csv` | Prediction for every test reaction (seed, reaction_id, y_true, y_pred) |
| `results/splits_70_30_seed0-9.csv` | The splits used (by reaction_id) |
| `results/validation_doyle_rf500_sqrt.csv` | R² on the Buchwald-Hartwig data with the fast setting (2048 bits, max_features=sqrt), ten splits, mean 0.927 |

### How to run

```bash
cd mff_benchmark
./setup_env.sh                      # python3 -m venv .venv && pip install -r requirements.txt
source .venv/bin/activate
python scripts/clean_shiomi_datasheets.py        # data/raw -> data/clean (already included)
python scripts/run_sbr_comparison.py             # four datasets x ten splits, about 20 min on an M2 Mac
python scripts/run_sbr_like_features_rf.py       # control models, a few minutes
```

## 3. Correspondence with the SI tables

| SI | Source | Script |
|---|---|---|
| Table S2 (15 pairwise comparisons, MAE), Table S3 (R²) | family `A_pairwise6_*` in `statistics/results/paired_wilcoxon_holm.csv` | `make_si_tables.py`, output `si_tables.md` |
| Table S4 (MAE and R² of all models, mean ± SD) | `statistics/input/all_splits_all_reactions.csv` + `mff_benchmark/results/*.csv` | same |
| Table S5 (four SBR representations versus MFF) | family `B_SBR_vs_MFF` in `paired_wilcoxon_holm.csv` | same |

All numbers quoted in the SI section "Comparison with MFF" (RedAm 3.94 versus 4.20, DeaminoCpl 7.99 versus 7.44, ElectroCpl 12.38 versus 11.90, UmAS 8.25 versus 8.36; controls 8.23, 8.60 and 3.84) are taken from Table S4 in `si_tables.md`.
