"""Reliability layer: how far can this patient's explanation be trusted?

For each patient we compute three numbers in [0, 1] and average them:

- agreement  Do the methods rank the important features the same way, with the
             same direction? Mean pairwise Spearman correlation of the signed
             attributions over the union of each pair's top-k features,
             mapped from [-1, 1] to [0, 1].
- overlap    Do the methods name the same top-k features? Mean pairwise
             Jaccard index of the top-k sets.
- stability  Does the explanation survive retraining? The model is retrained
             on patient-level bootstrap resamples; we compare this patient's
             attribution under each retrained model with the original, using
             the same rank measure as `agreement`.

This is the consistency requirement of Tonekaboni et al. [5]: an explanation
should not change because of an arbitrary technical choice (which method,
which random resample of the training data).
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

from . import config


def top_k(a, k=config.TOP_K):
    return set(np.argsort(-np.abs(a))[:k])


def rank_agreement(a, b, k=config.TOP_K):
    """Spearman correlation of two signed attribution vectors on the union of
    their top-k features, mapped to [0, 1]."""
    idx = sorted(top_k(a, k) | top_k(b, k))
    x, y = np.asarray(a)[idx], np.asarray(b)[idx]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.5
    r = spearmanr(x, y).statistic
    return 0.5 if np.isnan(r) else (r + 1) / 2


def jaccard(a, b, k=config.TOP_K):
    A, B = top_k(a, k), top_k(b, k)
    return len(A & B) / len(A | B)


def agreement_score(attrs: dict, k=config.TOP_K):
    pairs = list(combinations(attrs.values(), 2))
    return float(np.mean([rank_agreement(a, b, k) for a, b in pairs]))


def overlap_score(attrs: dict, k=config.TOP_K):
    pairs = list(combinations(attrs.values(), 2))
    return float(np.mean([jaccard(a, b, k) for a, b in pairs]))


def stability_score(base_attr, retrained_attrs, k=config.TOP_K):
    return float(np.mean([rank_agreement(base_attr, r, k) for r in retrained_attrs]))


def combine(agreement, overlap, stability, weights=config.RELIABILITY_WEIGHTS):
    w = weights
    return (w["agreement"] * agreement + w["overlap"] * overlap + w["stability"] * stability) / sum(w.values())


def band(score, bands=config.RELIABILITY_BANDS):
    if score >= bands["high"]:
        return "High"
    if score >= bands["moderate"]:
        return "Moderate"
    return "Low"


def normalise(a):
    m = np.max(np.abs(a))
    return a / m if m > 0 else a


class OODDetector:
    """How unusual is this patient compared with the training population?
    Isolation forest on one-hot features; returns a percentile in [0, 100],
    where 99 means more unusual than 99% of training patients."""

    def __init__(self, space, train_df, seed=config.SEED):
        self.space = space
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr = self._mat(train_df, fit=True)
        self.forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=4).fit(Xtr)
        self.train_scores = np.sort(-self.forest.score_samples(Xtr))

    def _mat(self, df, fit=False):
        X = self.space.frame(df)
        cat = X[self.space.categorical].astype(str)
        cat_m = self.enc.fit_transform(cat) if fit else self.enc.transform(cat)
        num = X[self.space.numeric].to_numpy(dtype=float)
        if fit:
            self.mu, self.sd = num.mean(0), num.std(0) + 1e-9
        return np.hstack([(num - self.mu) / self.sd, cat_m])

    def percentile(self, df):
        s = -self.forest.score_samples(self._mat(df))
        return 100 * np.searchsorted(self.train_scores, s) / len(self.train_scores)


@dataclass
class Assessment:
    probability: float
    probability_range: tuple
    attributions: dict
    agreement: float
    overlap: float
    stability: float
    score: float
    band: str
    ood_percentile: float
    features: list
    consensus: np.ndarray = field(default=None)
    sign_conflicts: list = field(default_factory=list)

    def top_features(self, n=5):
        order = np.argsort(-np.abs(self.consensus))[:n]
        return [(self.features[i], float(self.consensus[i])) for i in order]


class ReliabilityEngine:
    """Wraps a model, its explainers, retrained copies and an OOD detector,
    and produces an `Assessment` for one patient."""

    def __init__(self, model, explainers, stability_models, ood: OODDetector, k=config.TOP_K):
        self.model = model
        self.explainers = explainers
        self.stability_models = stability_models
        self.ood = ood
        self.k = k
        # Stability is measured with the model's own (exact, deterministic)
        # explainer, so changes come from retraining and not from sampling noise.
        self.stability_explainer = explainers[0]

    def attributions(self, x_codes):
        return {e.name: e.explain(self.model, x_codes) for e in self.explainers}

    def assess(self, x_codes) -> Assessment:
        x_codes = np.asarray(x_codes, dtype=float)
        df = self.model.space.decode(x_codes[None, :])
        attrs = self.attributions(x_codes)
        agr = agreement_score(attrs, self.k)
        ovl = overlap_score(attrs, self.k)
        base = attrs[self.stability_explainer.name]
        retrained = [self.stability_explainer.explain(m, x_codes) for m in self.stability_models]
        stab = stability_score(base, retrained, self.k) if retrained else float("nan")
        score = combine(agr, ovl, stab)

        p = float(self.model.predict_proba(df)[0])
        ps = [float(m.predict_proba(df)[0]) for m in self.stability_models] + [p]

        norm = np.vstack([normalise(a) for a in attrs.values()])
        consensus = norm.mean(axis=0)
        union = set().union(*(top_k(a, self.k) for a in attrs.values()))
        conflicts = [
            self.model.space.features[j] for j in sorted(union)
            if (np.sign(norm[:, j]) > 0).any() and (np.sign(norm[:, j]) < 0).any()
            and np.abs(norm[:, j]).max() > 0.2
        ]
        return Assessment(
            probability=p, probability_range=(min(ps), max(ps)), attributions=attrs,
            agreement=agr, overlap=ovl, stability=stab, score=score, band=band(score),
            ood_percentile=float(self.ood.percentile(df)[0]),
            features=self.model.space.features, consensus=consensus, sign_conflicts=conflicts,
        )
