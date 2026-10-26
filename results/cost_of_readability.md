# Cost of readability (test set, patient-level split)

Test set: 14,815 encounters, 10,499 patients, 11.10% readmitted within 30 days. All models calibrated on a separate set of patients.

| Model | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.6718 | 0.2140 | 0.0946 | 0.0142 |
| EBM (readable) | 0.6736 | 0.2131 | 0.0946 | 0.0105 |
| XGBoost (complex) | 0.6797 | 0.2355 | 0.0934 | 0.0069 |

Chance-level AUPRC equals the readmission rate (0.111).

## Paired patient-level bootstrap (1,000 resamples)

- XGBoost AUROC 95% CI: 0.6645 to 0.6959
- EBM AUROC 95% CI: 0.6582 to 0.6898
- **Cost of readability (XGBoost minus EBM): +0.0060 AUROC (95% CI -0.0011 to +0.0133, p = 0.106)**
- EBM minus logistic regression: +0.0018 AUROC (95% CI -0.0013 to +0.0049, p = 0.250)

Rudin's claim [2] is that readability usually costs little accuracy. A difference whose confidence interval sits near zero supports it on this task.