# What each file does

Open the files in this order. It follows the data from the raw CSV to the final report for one patient.

```
data.py → models.py → explainers.py → reliability.py → counterfactual.py → pipeline.py → run.py
                                                                              ↘ experiments/
```

## Starting point

### `run.bat`
Double-click launcher for Windows. The first time, it creates a Python environment (`.venv`) and installs the libraries from `requirements.txt`. Every time, it then runs `run.py`.

### `run.py`
The one file you run. It reads the options you type (`--patient`, `--model`, `--train` and so on), loads the trained system, and prints the report for a patient.
- `explain()` prints risk, reliability, main factors, each method's top factors, the "what if" result, and the real outcome.
- `main()` decides what to do. If the trained models are missing, it trains them first.

### `requirements.txt`
The list of Python libraries, with exact versions, so the results can be repeated on any computer.

## The `xaicdss` package (the actual system)

### `config.py`
All settings in one place: file paths, the random seed (42, so results repeat), the split sizes (70/15/15), how many LIME samples to use (1,000), how many retrained models to use for stability (5), and the High/Moderate/Low cut-offs (0.75 and 0.65).

### `data.py`: step 1, the data
- `load_raw()` reads `diabetic_data.csv`. Only `?` counts as missing, because "None" is a real value (test not done).
- `build_features()` cleans the data. It removes stays ending in death or hospice and turns the raw columns into 26 features: age, length of stay, past visits, diagnosis groups, medicines and so on. It also builds the target: readmitted within 30 days, yes or no.
- `_icd9_group()` turns thousands of diagnosis codes into 9 groups (heart, lungs, diabetes and so on).
- `FEATURE_DICTIONARY` gives each feature a plain-English name and says whether a clinician could change it before discharge.
- `FeatureSpace` converts text categories into numbers and back, because LIME and ablation need plain numbers.
- `patient_split()` splits the data by patient into train, calibration and test.
- `check_no_leakage()` stops the program if any patient is in two splits, or if an ID column is used as a feature.

### `models.py`: step 2, the models
- `make_ebm()` creates the readable model (Explainable Boosting Machine, one curve per feature).
- `make_xgb()` creates the complex model (XGBoost, 400 decision trees).
- `make_lr()` creates the logistic regression baseline.
- `train()` trains any of the three on the training patients.
- `ClinicalModel` is a wrapper that gives every model the same functions: `predict_proba()` for the risk and `logit()` for the raw score that the explanations use. `calibrate()` makes "20%" really mean 20%.
- `resample_patients()` makes a resampled copy of the training patients, used to retrain models for the stability check.
- `metrics()` calculates AUROC, AUPRC, Brier and ECE. `bootstrap_compare()` checks whether the gap between EBM and XGBoost is real. `subgroup_auroc()` checks fairness by race, gender and age.

### `explainers.py`: step 3, the explanations
All explainers have the same function, `explain(model, patient)`, which returns one number per feature: how much it pushed the risk up (+) or down (−).
- `TreeShapExplainer` gives SHAP values for XGBoost, using XGBoost's built-in exact method.
- `EBMTermExplainer` reads the EBM's own curve values for this patient.
- `LimeExplainer` runs LIME: 1,000 changed copies of the patient and a straight-line fit around them.
- `AblationExplainer` swaps each feature with values from 50 other patients and measures the drop in risk.
- `explainers_for()` picks the right set of 3 methods for the chosen model.

### `reliability.py`: step 4, the reliability score (our contribution)
- `rank_agreement()`: do two methods order the top factors the same way? (Spearman correlation, scaled to 0 to 1)
- `jaccard()`: how many of the top 5 factors two methods share.
- `agreement_score()`, `overlap_score()` and `stability_score()` average those checks across methods or retrained models.
- `combine()` is the final score, the average of the three. `band()` turns it into High, Moderate or Low.
- `OODDetector` uses an Isolation Forest to say how unusual a patient is (0 to 100).
- `ReliabilityEngine.assess()` does everything for one patient and returns an `Assessment`: risk, risk range, every method's explanation, the three checks, the score, the band, unusualness, and any factors where the methods disagree on direction.

### `counterfactual.py`: the "what if"
- `CANDIDATES` lists the only changes allowed: discharge destination, insulin dose and length of stay. It never suggests stopping a medicine.
- `search()` tries each allowed change, keeps the one that lowers risk most, and repeats up to 3 times.

### `pipeline.py`: puts it all together
- `run()` loads the data, splits it, trains the 3 models and 10 retrained copies (5 EBM, 5 XGBoost), builds the unusual-patient detector, writes the result tables, and saves everything to `artifacts/system.joblib`.
- `cost_of_readability()` writes the EBM vs XGBoost comparison table.
- `fairness()` writes performance by subgroup.
- `load_system()` and `engine_for()` load the saved system so that `run.py` doesn't retrain every time.

### `experiments/rq2_synthetic.py`: RQ2
Makes fake data where the true answer is known. It makes a useless feature more and more similar to a real one and measures how much credit each method wrongly gives it. Output: `results/rq2_synthetic.*`.

### `experiments/rq1_rq3.py`: RQ1 and RQ3
Scores 2,000 test patients, then tests whether the methods disagree (RQ1) and whether low reliability goes with wrong predictions or unusual patients (RQ3). Output: `results/rq1_rq3_*.md`, `results/rq3_*.png`, `results/patient_scores_*.csv`.

## Other folders

| Folder | What's in it |
|---|---|
| `data/raw/` | The original UCI dataset (CSV) |
| `artifacts/` | The trained system (`system.joblib`), created by `pipeline.py` |
| `results/` | Every table and chart the code produces |
| `tests/test_pipeline.py` | 6 automatic checks: no patient in two splits, the leakage check catches a planted leak, IDs aren't features, conversion back and forth works, and the score maths is right |
| `docs/` | This guide, the plain-language overview (`HOW_IT_WORKS.md`) and the model card |
