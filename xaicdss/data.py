"""Data layer: load the UCI Diabetes 130-US Hospitals data, clean it, build
features, and split it by patient.

The split is the riskiest step in the project (report, Table 5.2): if one
patient's encounters land in both train and test, every reported number is
inflated. `patient_split` groups by `patient_nbr`, and `check_no_leakage`
fails loudly if any patient or identifier crosses the split.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config

ID_COLS = ["encounter_id", "patient_nbr"]
TARGET = "readmitted_30d"

# Discharge codes meaning the patient died or went to hospice. These patients
# cannot be readmitted, so their label is not meaningful (Strack et al. 2014).
EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}

MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "n_diabetes_meds", "n_med_changes",
]
CATEGORICAL = [
    "race", "gender", "admission_type", "discharge_disposition",
    "admission_source", "medical_specialty", "diag_1_group", "diag_2_group",
    "diag_3_group", "max_glu_serum", "A1Cresult", "insulin", "metformin",
    "change", "diabetesMed",
]
FEATURES = NUMERIC + CATEGORICAL

# Plain-language names and whether a clinician could plausibly change the
# value before discharge. Used by the interface and the counterfactual search.
FEATURE_DICTIONARY = {
    "age_years": ("Age (years, decade midpoint)", False),
    "time_in_hospital": ("Length of stay (days)", True),
    "num_lab_procedures": ("Lab tests during stay", False),
    "num_procedures": ("Non-lab procedures during stay", False),
    "num_medications": ("Distinct medications given", False),
    "number_outpatient": ("Outpatient visits, past year", False),
    "number_emergency": ("Emergency visits, past year", False),
    "number_inpatient": ("Inpatient admissions, past year", False),
    "number_diagnoses": ("Diagnoses recorded", False),
    "n_diabetes_meds": ("Diabetes medications prescribed", False),
    "n_med_changes": ("Diabetes medication dose changes", False),
    "race": ("Race", False),
    "gender": ("Gender", False),
    "admission_type": ("Admission type", False),
    "discharge_disposition": ("Discharge destination", True),
    "admission_source": ("Admission source", False),
    "medical_specialty": ("Admitting specialty", False),
    "diag_1_group": ("Primary diagnosis group", False),
    "diag_2_group": ("Secondary diagnosis group", False),
    "diag_3_group": ("Additional diagnosis group", False),
    "max_glu_serum": ("Max glucose serum test", False),
    "A1Cresult": ("HbA1c test result", False),
    "insulin": ("Insulin at discharge", True),
    "metformin": ("Metformin at discharge", True),
    "change": ("Diabetes medication changed", True),
    "diabetesMed": ("On diabetes medication", True),
}


def pretty(feature: str) -> str:
    return FEATURE_DICTIONARY.get(feature, (feature, False))[0]


def _icd9_group(code) -> str:
    """Map an ICD-9 code to the diagnosis groups used by Strack et al. 2014."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def _admission_type(i):
    return {1: "Emergency", 2: "Urgent", 3: "Elective"}.get(i, "Other/Unknown")


def _discharge(i):
    if i == 1:
        return "Home"
    if i == 6:
        return "Home with health service"
    if i in (3, 4, 5, 22, 23, 24, 15, 27, 28, 29, 30):
        return "Care facility (SNF/rehab/other)"
    if i in (2, 9, 10, 12, 16, 17):
        return "Other hospital/inpatient"
    if i == 7:
        return "Left against advice"
    return "Other/Unknown"


def _admission_source(i):
    if i == 7:
        return "Emergency room"
    if i in (1, 2, 3):
        return "Referral"
    if i in (4, 5, 6, 10, 18, 22, 25, 26):
        return "Transfer"
    return "Other/Unknown"


TOP_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
    "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
    "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
]


def load_raw(path=config.DATA_RAW) -> pd.DataFrame:
    # "None" is a real category (test not done), so only "?" means missing.
    return pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw table and return ids + features + target."""
    d = raw.copy()
    d = d[~d["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    d = d[d["gender"].isin(["Male", "Female"])]

    out = pd.DataFrame(index=d.index)
    out["encounter_id"] = d["encounter_id"]
    out["patient_nbr"] = d["patient_nbr"]

    out["age_years"] = d["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures",
              "num_medications", "number_outpatient", "number_emergency",
              "number_inpatient", "number_diagnoses"]:
        out[c] = d[c].astype(float)
    meds = d[MEDS]
    out["n_diabetes_meds"] = (meds != "No").sum(axis=1).astype(float)
    out["n_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)

    out["race"] = d["race"].fillna("Missing")
    out["gender"] = d["gender"]
    out["admission_type"] = d["admission_type_id"].map(_admission_type)
    out["discharge_disposition"] = d["discharge_disposition_id"].map(_discharge)
    out["admission_source"] = d["admission_source_id"].map(_admission_source)
    spec = d["medical_specialty"].fillna("Missing")
    out["medical_specialty"] = spec.where(spec.isin(TOP_SPECIALTIES + ["Missing"]), "Other")
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = d[f"diag_{i}"].map(_icd9_group)
    for c in ["max_glu_serum", "A1Cresult", "insulin", "metformin", "change", "diabetesMed"]:
        out[c] = d[c]

    out[TARGET] = (d["readmitted"] == "<30").astype(int)
    return out.reset_index(drop=True)


@dataclass
class FeatureSpace:
    """Fixed feature order and category levels.

    Models take a DataFrame with pandas categoricals. Model-agnostic explainers
    (LIME, ablation) work on a plain float matrix, so categoricals are stored
    as integer codes there and decoded back before calling the model.
    """

    numeric: list
    categorical: list
    levels: dict = field(default_factory=dict)

    @property
    def features(self):
        return self.numeric + self.categorical

    @property
    def categorical_idx(self):
        return [self.features.index(c) for c in self.categorical]

    @classmethod
    def fit(cls, df: pd.DataFrame):
        levels = {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}
        return cls(list(NUMERIC), list(CATEGORICAL), levels)

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature columns with categoricals typed consistently."""
        X = df[self.features].copy()
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.levels[c])
        for c in self.numeric:
            X[c] = X[c].astype(float)
        return X

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        X = self.frame(X)
        arr = np.empty((len(X), len(self.features)), dtype=float)
        for j, c in enumerate(self.features):
            arr[:, j] = X[c].cat.codes if c in self.categorical else X[c].to_numpy()
        return arr

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        arr = np.atleast_2d(arr)
        cols = {}
        for j, c in enumerate(self.features):
            if c in self.categorical:
                codes = np.clip(np.rint(arr[:, j]).astype(int), 0, len(self.levels[c]) - 1)
                cols[c] = pd.Categorical.from_codes(codes, categories=self.levels[c])
            else:
                cols[c] = arr[:, j]
        return pd.DataFrame(cols, columns=self.features)


def patient_split(df: pd.DataFrame, seed: int = config.SEED):
    """Split so each patient is entirely in train, calibration, or test."""
    groups = df["patient_nbr"].to_numpy()
    test_frac = config.SPLIT["test"]
    calib_frac = config.SPLIT["calib"] / (1 - test_frac)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    rest_idx, test_idx = next(gss.split(df, groups=groups))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed + 1)
    tr, ca = next(gss2.split(rest, groups=rest["patient_nbr"].to_numpy()))
    splits = {
        "train": rest.iloc[tr].reset_index(drop=True),
        "calib": rest.iloc[ca].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }
    check_no_leakage(splits)
    return splits


class LeakageError(AssertionError):
    pass


def check_no_leakage(splits: dict, feature_names=FEATURES):
    """Raise LeakageError if any patient or encounter crosses splits, or if an
    identifier or the target is used as a feature."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["patient_nbr"]) & set(splits[b]["patient_nbr"])
            if shared:
                raise LeakageError(f"{len(shared)} patients appear in both {a} and {b}")
            shared_enc = set(splits[a]["encounter_id"]) & set(splits[b]["encounter_id"])
            if shared_enc:
                raise LeakageError(f"{len(shared_enc)} encounters appear in both {a} and {b}")
    banned = set(ID_COLS) | {TARGET, "readmitted"}
    bad = banned & set(feature_names)
    if bad:
        raise LeakageError(f"Identifier or target used as a feature: {sorted(bad)}")
    return True


def leakage_report(df: pd.DataFrame, splits: dict) -> str:
    """Text summary for results/: what a row-level split would have leaked."""
    n_pat = df["patient_nbr"].nunique()
    multi = (df.groupby("patient_nbr").size() > 1).sum()
    rng = np.random.default_rng(config.SEED)
    row_test = rng.random(len(df)) < config.SPLIT["test"]
    leaked = len(set(df.loc[row_test, "patient_nbr"]) & set(df.loc[~row_test, "patient_nbr"]))
    lines = [
        "# Data split and leakage check",
        "",
        f"- Encounters after cleaning: {len(df):,} (patients: {n_pat:,})",
        f"- Patients with more than one encounter: {multi:,}",
        f"- 30-day readmission rate: {df[TARGET].mean():.2%}",
        "",
        "| Split | Encounters | Patients | Readmitted <30d |",
        "|---|---|---|---|",
    ]
    for k, s in splits.items():
        lines.append(f"| {k} | {len(s):,} | {s['patient_nbr'].nunique():,} | {s[TARGET].mean():.2%} |")
    lines += [
        "",
        "Patient overlap between any two splits: **0** (enforced by `check_no_leakage`).",
        "",
        f"For comparison, a naive row-level split would have put {leaked:,} patients in "
        "both the training and test data.",
    ]
    return "\n".join(lines)
