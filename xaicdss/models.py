"""Model layer: a readable model (EBM), a complex model (XGBoost), and a
logistic-regression baseline, all trained on the same patient-level split and
calibrated on a separate set of patients.

Every model is wrapped in `ClinicalModel`, which exposes the only three things
the explanation layer needs (report, Section 4.2): the model, the input data,
and the prediction.
"""

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config
from .data import TARGET, FeatureSpace


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ClinicalModel:
    """A fitted model plus Platt calibration.

    `logit` is the model's raw log-odds (what attributions explain).
    `predict_proba` is the calibrated risk shown to the clinician. Platt scaling
    is monotone, so it never changes which features matter or in which order.
    """

    def __init__(self, name, estimator, space: FeatureSpace, kind):
        self.name = name
        self.estimator = estimator
        self.space = space
        self.kind = kind  # "ebm", "xgb" or "lr"
        self.calibrator = None

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        X = self.space.frame(X)
        if self.kind == "xgb":
            return self.estimator.predict(X, output_margin=True)
        if self.kind == "ebm":
            return self.estimator.decision_function(X)
        return self.estimator.decision_function(X)

    def logit_codes(self, arr: np.ndarray) -> np.ndarray:
        return self.logit(self.space.decode(arr))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = self.logit(X)
        if self.calibrator is None:
            return 1 / (1 + np.exp(-z))
        return self.calibrator.predict_proba(z.reshape(-1, 1))[:, 1]

    def proba_codes(self, arr: np.ndarray) -> np.ndarray:
        return self.predict_proba(self.space.decode(arr))

    def calibrate(self, X: pd.DataFrame, y):
        self.calibrator = LogisticRegression(C=1e6).fit(self.logit(X).reshape(-1, 1), y)
        return self


def make_xgb(seed=config.SEED, n_estimators=400):
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, tree_method="hist", enable_categorical=True,
        max_cat_to_onehot=1, eval_metric="logloss", random_state=seed, n_jobs=4,
    )


def make_ebm(seed=config.SEED, interactions=0):
    # Main effects only by default, so every term is one readable graph per
    # feature (Caruana et al. [1] also report the pairwise variant).
    return ExplainableBoostingClassifier(
        interactions=interactions, outer_bags=8, max_bins=128,
        random_state=seed, n_jobs=4,
    )


def make_lr(space: FeatureSpace):
    pre = ColumnTransformer([
        ("num", StandardScaler(), space.numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), space.categorical),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def train(kind: str, train_df: pd.DataFrame, space: FeatureSpace, seed=config.SEED, **kw):
    X, y = space.frame(train_df), train_df[TARGET].to_numpy()
    if kind == "xgb":
        est, name = make_xgb(seed, **kw), "XGBoost (complex)"
    elif kind == "ebm":
        est, name = make_ebm(seed, **kw), "EBM (readable)"
    elif kind == "lr":
        est, name = make_lr(space), "Logistic regression (baseline)"
    else:
        raise ValueError(kind)
    est.fit(X, y)
    return ClinicalModel(name, est, space, kind)


def resample_patients(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Bootstrap by patient: draw patients with replacement, keep all their
    encounters. Used for the stability measure and the confidence range."""
    rng = np.random.default_rng(seed)
    pats = df["patient_nbr"].unique()
    pick = rng.choice(pats, size=len(pats), replace=True)
    counts = pd.Series(pick).value_counts()
    idx = df.reset_index(drop=True).groupby("patient_nbr").indices
    rows = np.concatenate([np.repeat(idx[p], c) for p, c in counts.items()])
    return df.iloc[rows].reset_index(drop=True)


# ---------------------------------------------------------------- evaluation

def ece(y, p, bins=10):
    """Expected calibration error with equal-frequency bins."""
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        total += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return total / len(y)


def metrics(y, p):
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "ECE": ece(y, p),
    }


def bootstrap_compare(y, p_a, p_b, groups, n_boot=1000, seed=config.SEED):
    """Patient-level paired bootstrap of AUROC for two models.
    Returns CIs for each AUROC and for the difference (a - b)."""
    rng = np.random.default_rng(seed)
    pats = np.unique(groups)
    idx = pd.Series(np.arange(len(y))).groupby(groups).indices
    a, b = [], []
    for _ in range(n_boot):
        rows = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        if y[rows].min() == y[rows].max():
            continue
        a.append(roc_auc_score(y[rows], p_a[rows]))
        b.append(roc_auc_score(y[rows], p_b[rows]))
    a, b = np.array(a), np.array(b)
    d = a - b
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return {
        "a_ci": ci(a), "b_ci": ci(b), "diff": d.mean(), "diff_ci": ci(d),
        "p_two_sided": 2 * min((d <= 0).mean(), (d >= 0).mean()),
    }


def subgroup_auroc(df, p, cols=("race", "gender", "age_years")):
    """AUROC and calibration by subgroup, for the fairness section of the model card."""
    rows = []
    y = df[TARGET].to_numpy()
    for c in cols:
        vals = df[c]
        if c == "age_years":
            vals = pd.cut(df[c], [0, 40, 60, 80, 120], labels=["<40", "40-59", "60-79", "80+"])
        for v in pd.Series(vals).dropna().unique():
            m = (vals == v).to_numpy()
            if m.sum() < 100 or y[m].min() == y[m].max():
                continue
            rows.append({
                "attribute": c, "group": str(v), "n": int(m.sum()),
                "readmit_rate": y[m].mean(), "mean_predicted": p[m].mean(),
                "AUROC": roc_auc_score(y[m], p[m]),
            })
    return pd.DataFrame(rows)
