# How the system works (plain-language guide)

Use this to explain the project in a review, a viva or at the exhibition. It follows one patient through the system, then covers the experiments.

## The one-line idea

Most AI health systems show a prediction and an explanation. **Ours also tells you how much to trust the explanation**, by checking whether different explanation methods tell the same story for that patient.

## The example from our report

A nurse is discharging a diabetic patient. The system says **68% chance of readmission within 30 days**. What should it show next to that number, and how far can she trust it?

For each patient the system prints four things:

1. **Risk**: the predicted chance, and the range we get when we retrain the model on slightly different data (for example "51% to 71%"). A narrow range means the number itself is stable.
2. **Main factors**: what pushed the risk up or down, such as "13 inpatient admissions in the past year raises risk".
3. **Reliability score (0 to 1)**: how consistent that explanation is: High, Moderate or Low. If it is Low, a warning says to rely more on clinical judgement.
4. **What if?**: which adjustable factors (discharge destination, insulin dose, length of stay) would change the model's risk most. This is not treatment advice.

A clinician dashboard comes in a later phase. For now the system runs from the terminal.

## Where do the "patients" come from?

We have no real hospital patients, and we don't need any. The dataset is split by patient:

- **70% training patients**: the models learn from these.
- **15% calibration patients**: used to make the risk percentages honest.
- **15% test patients (14,815 hospital stays)**: the models **never see these during training**.

The test patients play the role of "new patients arriving at the hospital". When you run `python run.py --patient 3923`, the system takes test patient 3923, predicts their risk, explains it and scores the explanation. Because these are real historical records, we also know what actually happened, so we can check whether the system was right. That is how every research paper in this area evaluates a model, and it is more honest than typing in made-up patient values.

## The pipeline, step by step

```
raw hospital data → clean + split by patient → train 2 models → explain with 3 methods
      → compare the explanations → reliability score → printed report per patient
```

### Step 1: Data (`xaicdss/data.py`)
- UCI Diabetes 130-US Hospitals: 101,766 hospital stays from 130 US hospitals, 1999-2008. It is public and de-identified.
- We remove stays that ended in death or hospice (those patients cannot come back), and drop `weight` because 97% of it is missing.
- **The key safety step:** many patients appear more than once. If the same patient is in both training and testing, the model has "seen the answer" and looks better than it is. We split by **patient**, not by row. A test (`tests/test_pipeline.py`) fails automatically if any patient is in two sets. A naive row split would have leaked 5,395 patients.
- Only 11.4% of stays end in readmission, so we report AUROC, AUPRC and Brier score, not plain accuracy.

### Step 2: Two models (`xaicdss/models.py`)
- **EBM (Explainable Boosting Machine)**: the *readable* model. It adds up one simple graph per feature, like the model Caruana et al. used to find the asthma problem.
- **XGBoost**: the *complex* model, hundreds of decision trees.
- Both are trained on exactly the same patients and calibrated so that "20%" really means about 20 in 100.

**Result (cost of readability):** EBM AUROC 0.674, XGBoost 0.680. The difference (0.006) is not statistically significant (95% CI -0.001 to 0.013). **Readability costs almost nothing here**, which supports Rudin's argument.

### Step 3: Three explanation methods (`xaicdss/explainers.py`)
Each method answers "how much did each feature push this patient's risk up or down?"

| Method | Idea in one line |
|---|---|
| **SHAP** (XGBoost) / **EBM terms** (EBM) | Game theory: share the prediction fairly between features. For EBM, the model's own graphs give this directly. |
| **LIME** | Make 1,000 slightly changed copies of the patient, see how the risk changes, fit a simple line around this patient. |
| **Ablation** | Replace one feature at a time with values from other patients and see how much the risk drops. |

### Step 4: The reliability score (`xaicdss/reliability.py`), our contribution
Three checks, each from 0 to 1, averaged:

- **Agreement**: do the methods rank the important features in the same order and direction?
- **Overlap**: of each method's top 5 features, how many are shared?
- **Stability**: we retrain the model 5 times on resampled patients. Does this patient's explanation stay the same?

This turns Tonekaboni et al.'s requirement ("an explanation should not change because of an arbitrary technical choice") into a number for every patient.

We also compute **how unusual** the patient is compared with the training data (0 to 100).

## The three research questions and what we found

**RQ1: Do the methods disagree?** Yes, often. On 2,000 test patients (XGBoost), the methods' top-5 factor lists overlap by only 43% on average, and no patient got identical top-5 lists from all three methods. Rank agreement is higher (0.81), so the methods usually agree on the biggest factors and disagree on the rest. For the readable EBM the methods agree more (overlap 53%, reliability 0.77 vs 0.71).

**RQ2: Why do they disagree? (synthetic data)** We made artificial data where we know the truth. Feature x1 has **no** effect but we made it more and more similar to a real feature, x0. As the correlation goes from 0 to 0.99, the share of credit wrongly given to x1 rises from about 1% to 14% (EBM) and from 4% to 12% (XGBoost), and accuracy falls. **But the methods keep agreeing with each other** (agreement stays between 0.91 and 0.96 at every level). So when features are very similar, the methods are wrong *together*. Agreement cannot catch this kind of error, which is an important honest limitation.

**RQ3: Is low reliability a warning sign?** Not for errors, but yes for unusual patients.
- For XGBoost, reliability was almost the same for right and wrong predictions (0.714 vs 0.711, p = 0.18). As an error detector it is close to chance (AUROC 0.52, where 0.5 is chance). Only the most reliable fifth of patients stood out, with a lower error rate (18.5% against about 24% for the rest).
- In an earlier run with a different LIME random seed this result looked significant (p = 0.002). That it disappears with a new seed tells us the link to errors is **not robust**, which is worth saying openly.
- The clear and repeatable result: reliability is lower for **unusual patients**, those unlike the training data (correlation -0.28, p < 0.001, the same in every run). The most reliable fifth had an average unusualness of 39/100, the least reliable 63/100.
- For the **readable EBM** there is no warning signal either. Its explanations are already more consistent, and its most reliable patients are also its highest-risk ones, so error rates simply follow risk.

**Bottom line:** the reliability score tells you when an explanation is shaky and when a patient is unusual, but it does not tell you when the prediction itself is wrong. That is the honest answer to RQ3, and our report (risk register) said a negative result is still a real result: it tells hospitals not to treat explanation disagreement as proof that a prediction is wrong.

## Likely viva questions

- **Why only AUROC 0.68 when reviews report a median of 0.87?** Readmission is hard to predict from hospital records alone because it depends on things after discharge (home support, follow-up). Published work on this dataset reports similar numbers. Splitting by patient also removes the inflated scores that row splits give.
- **Why not just use the readable model?** Here you could, because it costs almost no accuracy. That is one of our findings. The reliability layer still matters for any complex model a hospital already uses.
- **Isn't LIME random?** Yes, and that randomness is one source of disagreement the score is designed to expose.
- **Why patient-level split?** See Step 1. It is the biggest risk in our risk register.
- **Is "what if" a treatment recommendation?** No. It shows what the model reacts to. Treatment recommendations are out of scope.

## How to run it

Double-click `run.bat` (Windows), or from a terminal inside the folder:

```bash
python run.py                    # explain the two demo patients
python run.py --patient 3923     # explain any test patient (0 to 14814)
python run.py --random           # a random test patient
python run.py --model ebm        # use the readable EBM instead of XGBoost
```

## Suggested demo (3 minutes)

1. **`python run.py --patient 3923`**. Risk 68%, like the nurse in our report. Reliability is High, the factors make clinical sense (13 inpatient admissions last year, glucose over 300), but the "unusual patient" warning shows this person is unlike most of the training data. No adjustable factor lowers the risk: the model's concern is mostly history, not anything decided at discharge.
2. **`python run.py --patient 10613`**. Risk only 5%, but reliability is **Low** and the warning appears: the methods disagree on the direction of age and number of medications. This patient was in fact readmitted. Use it to explain why a confident-looking number with a shaky explanation deserves a second look, then be honest that across 2,000 patients the score does not reliably predict errors (RQ3).
3. **`python run.py --patient 3923 --model ebm`**. The same patient explained by the readable model, for comparison.
4. Open `results/` to show the cost-of-readability table and the experiment results and charts.
