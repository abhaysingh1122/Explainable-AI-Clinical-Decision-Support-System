# Model card: 30-day readmission risk (XAI-CDSS prototype)

## What it is for
Predicts the chance that an adult diabetic inpatient is readmitted within 30 days of discharge, and shows how far the explanation of that prediction can be trusted. **Research and teaching only. Not validated for clinical use.**

## Models
| | EBM (readable) | XGBoost (complex) |
|---|---|---|
| Type | Explainable Boosting Machine, main effects only (one graph per feature) | Gradient-boosted trees, 400 trees, depth 5 |
| Test AUROC | 0.674 | 0.680 |
| Test AUPRC (chance = 0.111) | 0.213 | 0.236 |
| Brier score | 0.0946 | 0.0934 |
| Calibration error (ECE) | 0.011 | 0.007 |

Both calibrated with Platt scaling on a separate set of patients. A logistic regression baseline scores AUROC 0.672.

## Data
UCI Diabetes 130-US Hospitals, 1999-2008 (CC BY 4.0). 99,340 encounters from 69,987 patients after cleaning. Split 70/15/15 **by patient** into train, calibration and test sets; an automatic test fails if any patient appears in two sets.

26 features: age, length of stay, counts of lab tests, procedures, medications, prior outpatient/emergency/inpatient visits, diagnoses, diagnosis groups, admission and discharge information, glucose and HbA1c tests, insulin and metformin status, medication changes, race and gender.

Removed: `weight` (97% missing), `payer_code` (administrative), patient and encounter IDs, and encounters ending in death or hospice.

## Performance across groups (test set, AUROC)
| Group | EBM | XGBoost |
|---|---|---|
| Female / Male | 0.676 / 0.671 | 0.682 / 0.677 |
| Caucasian / African American | 0.671 / 0.675 | 0.681 / 0.668 |
| Age under 40 | 0.781 | 0.797 |
| Age 80 and over | 0.587 | 0.610 |

Performance is similar across gender and the two largest race groups. It is **noticeably worse for patients aged 80 and over**, which should be stated whenever the system is shown. Smaller race groups have too few patients for firm conclusions. Full table: `results/subgroup_performance.csv`.

## Limitations
- Discrimination is modest (AUROC about 0.68), which is typical for this dataset; 30-day readmission depends on many things not in the record (social support, follow-up care).
- The data is 1999-2008 and from US hospitals only. It has not been tested on any other hospital.
- The reliability score only measures whether explanations are *consistent*. Our synthetic experiment shows methods can agree with each other and still be wrong when two features are strongly correlated.
- "What if" suggestions describe the model's sensitivity, not treatment advice.

## Human oversight
The clinician makes every decision; the system only gives a risk, an explanation and a reliability score. A clinician interface with a way to record disagreement is planned for a later phase (report objective 7).
