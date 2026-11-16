# SBR2-small-data-yield-prediction

Structure-Based Regression (SBR) for reaction yield prediction from small substrate-scope datasets using molecular fingerprints and ElasticNet regression.

This repository accompanies the manuscript:

**Reaction-Dependent Predictability of Yields from Small Substrate-Scope Datasets**

Shinya Shiomi, Takuma Morimoto, and Masahiro Yoshida

## Overview

How much chemical information is needed to predict reaction yields from small experimental datasets?

This study evaluates whether simple molecular fingerprints and linear regression can provide useful yield predictions without explicitly incorporating reaction conditions or quantum chemical descriptors.

Four SBR representations were systematically evaluated across four reaction datasets. Predictive performance was compared against constant-yield baseline models (Null mean and Null median), and learning curve analyses were performed to examine the effects of training dataset size.

The results highlight both the potential and reaction-dependent limitations of structure-based yield prediction in small-data settings.

## Reaction datasets

- **RedAm:** Reductive amination (118 reactions)
- **DeaminoCpl:** Deaminative coupling (96 reactions)
- **ElectroCpl:** Electrochemical coupling (97 reactions)
- **UmAS:** Umpolung amide synthesis and related oxidative esterification (190 reactions collected from eight publications)

## Molecular representations

Morgan fingerprints (radius = 2, 2048 bits) were used to construct four representations:

- **SBR-C2:** Concatenation of two reactant fingerprints (4096 dimensions)
- **SBR-2:** Element-wise addition of two reactant fingerprints (2048 dimensions)
- **SBR-2P:** Concatenation of the summed reactant fingerprints and the product fingerprint (4096 dimensions)
- **SBR-3:** Element-wise addition of two reactant fingerprints and the product fingerprint (2048 dimensions)

All representations were evaluated using ElasticNet regression.

## Contents

The associated analysis includes:

- Molecular fingerprint generation
- ElasticNet modeling with cross-validation
- Evaluation over ten random train–test splits
- Comparison against Null mean and Null median models
- Learning curve analysis
- Reaction datasets and prediction results

## Requirements

- Python 3.10
- RDKit
- scikit-learn
- pandas
- NumPy
- SciPy
- matplotlib

The software versions used for the main analysis are reported in the Supporting Information.


## Usage
Download or clone this repository and open a terminal in its root directory.

By default, the scripts are configured to analyze the UmAS dataset (`ML_UmAS.csv`). To analyze another reaction dataset, modify `INPUT_CSV` in Code S1 and `DATASET` in Codes S2 and S3 accordingly.

Run the scripts in the following order.

Select the same dataset in Code S1, Code S2, and Code S3 before running the scripts.

### 1. Generate molecular fingerprints

```bash
python scripts/SBR_fingerprint_generation.py
```

This generates four feature matrices in `SBR_features/`.

### 2. Evaluate SBR models

```bash
python scripts/SBR_model_evaluation.py
```

This evaluates four SBR representations and two null baselines across ten matched 70:30 train-test splits.

Results are saved in `SBR_model_results/`.

### 3. Generate learning curves

```bash
python scripts/SBR_learning_curve.py
```

This generates nested-subset learning curves for SBR-2 and the null models.

Results are saved in `SBR_learning_curve_results/`.

**Note:** The full learning curve analysis is computationally intensive. Quick-test settings are provided as comments in Code S3.


## Reproducibility

The main model evaluations used ten random 70:30 train–test splits. ElasticNet hyperparameters were optimized by five-fold cross-validation within each training set.

Learning curve analyses used ten random 80:20 train–test splits, with repeated training-subset sampling.

Additional methodological details, statistical analyses, and the comparison with the multiple fingerprint features (MFF) approach are described in the Supporting Information.

## Publication

Shiomi, S.; Morimoto, T.; Yoshida, M.

*Reaction-Dependent Predictability of Yields from Small Substrate-Scope Datasets.*

Manuscript in preparation.

## Notes

This repository is intended to support reproducibility and further investigation of structure-based reaction yield prediction using small experimental datasets.
