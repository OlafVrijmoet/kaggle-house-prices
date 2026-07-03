# House Prices: Advanced Regression

Predicting sale prices for houses in Ames, Iowa (1,460 training rows, 79 features) for the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Scored on RMSE between the log of the predicted and the log of the actual price, so errors count relative to the price of the house.

**Result: public leaderboard score improved from 0.12909 to 0.12230 across 11 iterations in six days.** Two of those iterations shipped nothing, on purpose: the validation harness rejected the candidate features, and both times the next, narrower idea won instead.

## Results

| # | Configuration | Local validation | Public LB |
|---|---|---|---|
| 1 | XGBoost, tuned on a single split | n/a | 0.12909 |
| 2 | Same model re-selected under repeated-split validation | n/a | 0.13184 |
| 3 | 50/50 XGBoost + CatBoost blend | n/a | 0.12643 |
| 4 | + structural features (two rounds: sizes, ratios, ages) | n/a | 0.12486 |
| 5 | + seed bagging, 5 seeds equal weight | n/a | 0.12388 |
| 6 | Tuning day: submission blends, extra model diversity, Optuna (9 submissions) | n/a | 0.12384 |
| 7 | + ordinal encodings for 22 quality-scale columns | 0.1074 | 0.12280 |
| 8 | Broad additive round 2: subsystem totals, quality-size, effective age | 0.1082, lost to baseline | not submitted |
| 9 | Asymmetric views: XGBoost drops the raw ordered categoricals, CatBoost keeps them | 0.1071 | 0.12262 |
| 10 | Deeper XGBoost cleanups, one family at a time (3 variants) | all lost to baseline | not submitted |
| 11 | + `log1p` copies of LotArea, LotFrontage, GrLivArea and total SF, XGBoost side only | 0.1070 | **0.12230** |

Local validation is the mean log RMSE over 5 fixed stratified 80/20 splits. Iterations 1 to 6 predate the surviving notebooks; their public scores are the recorded Kaggle submissions, and rows 7 to 11 are fully reproducible from the notebooks in this repo. The headline stat of the project sits between rows 5 and 11: a full day of tuning and blending (row 6, nine submissions) improved the public score by 0.00004, while the three feature-representation sprints that followed improved it by 0.00158.

## Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/00_eda.ipynb` | EDA ending in the concrete consequences for the setup: metric, outlier policy, imputation, encoding and transform candidates |
| `notebooks/01_feature_engineering_sprint.ipynb` | Four feature families under one harness; quality ordinals win (row 7) |
| `notebooks/02_feature_engineering_round2_sprint.ipynb` | Three additive round-2 families; all rejected, nothing shipped (row 8) |
| `notebooks/03_feature_replacement_sprint.ipynb` | Model-specific feature views; XGBoost drops raw ordered categoricals (row 9) |
| `notebooks/04_feature_cleanup_sprint.ipynb` | Deeper XGBoost-only cleanups; all rejected, nothing shipped (row 10) |
| `notebooks/05_feature_skew_revisit_sprint.ipynb` | Focused skew handling: four `log1p` features take the final step (row 11) |

Every sprint notebook runs the same protocol: build candidate feature paths, scout all of them on 3 outer splits with a fixed bag5 XGBoost + CatBoost blend, re-validate the shortlist on 5 fixed splits ranked by mean outer log RMSE, re-check the bagging variants on the winner, and export versioned submission files.

## Reproducing

1. Download the competition data into `data/` with the Kaggle CLI:
   `kaggle competitions download -c house-prices-advanced-regression-techniques -p data --unzip`
2. Create the environment: `conda env create -f environment.yml`
3. Run the notebooks in order. Each one is self-contained (features are rebuilt from the raw CSVs) and writes its submission files to `submissions/`.

Everything runs on a laptop CPU; a full sprint notebook takes roughly half an hour.
