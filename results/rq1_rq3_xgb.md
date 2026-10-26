# RQ1 and RQ3 results, XGB model (2,000 test patients)

## RQ1: do the methods disagree?

| Measure | Mean | 10th pct | Median | 90th pct |
|---|---|---|---|---|
| agreement | 0.814 | 0.672 | 0.835 | 0.929 |
| overlap | 0.429 | 0.263 | 0.448 | 0.587 |
| stability | 0.898 | 0.820 | 0.912 | 0.960 |
| reliability | 0.714 | 0.627 | 0.718 | 0.794 |

Top-5 features named by every method identically: 0.0% of patients. Average top-5 overlap (Jaccard): 0.43.

## RQ3: is low reliability a warning sign?

Wrong = patient flagged (top 20% risk, p >= 0.159) but not readmitted, or not flagged but readmitted.

- Mean reliability when wrong: 0.711; when right: 0.714 (Mann-Whitney p = 0.18)
- AUROC of low reliability for spotting wrong predictions: 0.521 (0.5 = no signal)
- Spearman correlation, reliability vs squared error: +0.061 (p = 0.0061)
- Spearman correlation, reliability vs how unusual the patient is: -0.277 (p = 1.7e-36)

| Reliability fifth | Patients | Error rate | Mean risk | Unusualness (pct) |
|---|---|---|---|---|
| Q1 (lowest) | 400 | 23.5% | 0.107 | 63 |
| Q2 | 400 | 25.5% | 0.117 | 55 |
| Q3 | 400 | 25.5% | 0.119 | 52 |
| Q4 | 400 | 24.0% | 0.120 | 46 |
| Q5 (highest) | 400 | 18.5% | 0.120 | 39 |

Note: error rate depends strongly on predicted risk (patients near the threshold are the ones that get misclassified), so compare the mean-risk column before reading the error column as an effect of reliability alone.