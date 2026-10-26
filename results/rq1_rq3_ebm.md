# RQ1 and RQ3 results, EBM model (2,000 test patients)

## RQ1: do the methods disagree?

| Measure | Mean | 10th pct | Median | 90th pct |
|---|---|---|---|---|
| agreement | 0.875 | 0.781 | 0.890 | 0.943 |
| overlap | 0.528 | 0.389 | 0.508 | 0.619 |
| stability | 0.920 | 0.845 | 0.934 | 0.973 |
| reliability | 0.774 | 0.708 | 0.774 | 0.837 |

Top-5 features named by every method identically: 0.2% of patients. Average top-5 overlap (Jaccard): 0.53.

## RQ3: is low reliability a warning sign?

Wrong = patient flagged (top 20% risk, p >= 0.150) but not readmitted, or not flagged but readmitted.

- Mean reliability when wrong: 0.786; when right: 0.771 (Mann-Whitney p = 2.2e-07)
- AUROC of low reliability for spotting wrong predictions: 0.422 (0.5 = no signal)
- Spearman correlation, reliability vs squared error: +0.252 (p = 2.5e-30)
- Spearman correlation, reliability vs how unusual the patient is: -0.060 (p = 0.0075)

| Reliability fifth | Patients | Error rate | Mean risk | Unusualness (pct) |
|---|---|---|---|---|
| Q1 (lowest) | 400 | 20.5% | 0.099 | 57 |
| Q2 | 400 | 21.0% | 0.104 | 49 |
| Q3 | 400 | 19.5% | 0.112 | 49 |
| Q4 | 400 | 26.5% | 0.123 | 49 |
| Q5 (highest) | 400 | 34.0% | 0.150 | 51 |

Note: error rate depends strongly on predicted risk (patients near the threshold are the ones that get misclassified), so compare the mean-risk column before reading the error column as an effect of reliability alone.