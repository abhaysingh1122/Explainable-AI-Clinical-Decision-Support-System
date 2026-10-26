# Data split and leakage check

- Encounters after cleaning: 99,340 (patients: 69,987)
- Patients with more than one encounter: 16,341
- 30-day readmission rate: 11.39%

| Split | Encounters | Patients | Readmitted <30d |
|---|---|---|---|
| train | 69,636 | 48,990 | 11.43% |
| calib | 14,889 | 10,498 | 11.50% |
| test | 14,815 | 10,499 | 11.10% |

Patient overlap between any two splits: **0** (enforced by `check_no_leakage`).

For comparison, a naive row-level split would have put 5,395 patients in both the training and test data.