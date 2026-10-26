# XAI-CDSS: Explainable AI for Clinical Decision Support

B.Tech major project, IILM University (2026-27): *An Explainable AI Framework for Clinical Decision Support Systems*.

The system predicts whether a diabetic patient will be **readmitted to hospital within 30 days**, explains the prediction with several methods, and gives a **reliability score** that says how much those explanations can be trusted for that patient.

> Research prototype for demonstration only. Do not use with real patients.

For a plain-language walkthrough you can present from, see **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**.

## Quick start

Needs Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m xaicdss.pipeline                         # 1. data, split, models (about 6 min)
python -m xaicdss.experiments.rq2_synthetic        # 2. similar-features experiment (about 2 min)
python -m xaicdss.experiments.rq1_rq3 --model xgb  # 3. agreement and error test (about 10 min)
python -m xaicdss.experiments.rq1_rq3 --model ebm
pytest -q                                          # tests, including the leakage test

streamlit run app/streamlit_app.py                 # 4. the interface
```

The trained system (`artifacts/system.joblib`) and all results are already included, so you can skip straight to step 4. If the app fails to load the saved models (usually a library version mismatch), run step 1 again to rebuild them.

## Project layout

```
data/raw/diabetic_data.csv   UCI Diabetes 130-US Hospitals (1999-2008), CC BY 4.0
xaicdss/
  data.py            cleaning, features, patient-level split, leakage check
  models.py          EBM (readable), XGBoost (complex), logistic regression, calibration
  explainers.py      SHAP, LIME, EBM terms, ablation behind one interface
  reliability.py     agreement, overlap, stability -> reliability score; unusual-patient detector
  counterfactual.py  "what if" search over adjustable factors
  pipeline.py        runs everything and saves artifacts/system.joblib
  experiments/
    rq1_rq3.py       do methods disagree? does disagreement warn of errors?
    rq2_synthetic.py what happens when two features are very similar?
app/streamlit_app.py the clinician interface
tests/               automatic tests (leakage, encoding, score maths)
results/             every table and figure the experiments produce
docs/                plain-language guide and model card
```

## Headline results

See `results/` for the full tables.

| Question | Result |
|---|---|
| Cost of readability | EBM AUROC 0.674 vs XGBoost 0.680. Difference 0.006, 95% CI -0.001 to 0.013, not significant |
| RQ1: do methods disagree? | Yes. Top-5 factor lists overlap only 43% (XGBoost) and 53% (EBM) on average |
| RQ2: similar features | Credit wrongly given to a fake, correlated feature rises from about 1% to 14% as correlation goes 0 to 0.99, but the methods keep agreeing with each other |
| RQ3: is disagreement a warning? | Not for errors: reliability is almost the same for right and wrong predictions (XGBoost p = 0.18, AUROC 0.52). But it is clearly lower for unusual patients (r = -0.28, p < 0.001) |

## Data

The UCI Diabetes 130-US Hospitals dataset: 101,766 encounters, 71,518 patients. After removing encounters that ended in death or hospice (they cannot be readmitted) and 3 records with unknown gender, 99,340 encounters remain. `weight` is dropped (97% missing). The target is readmission within 30 days (11.4%).
