"""Explanation layer: several attribution methods behind one interface.

Every explainer answers the same question for one patient: how much did each
feature push this prediction up or down? It returns one signed number per
feature, in the feature order of `FeatureSpace`. Methods differ in scale (log-
odds vs probability), so the reliability layer compares ranks and sets, never
raw magnitudes.

Methods
- TreeSHAP   exact Shapley values for XGBoost, in log-odds [4]
- EBM terms  the readable model's own additive contributions, in log-odds [1].
             For an additive model these are its exact Shapley values.
- LIME       local linear surrogate fitted on perturbed copies of the patient [3]
- Ablation   how much the log-odds drop when the feature is replaced by values
             from other patients (a one-feature-at-a-time counterfactual)

Counterfactual search is in `counterfactual.py`; it answers a different
question (what would have to change), so it is not an attribution.
"""

import warnings

import numpy as np
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer

from . import config
from .models import ClinicalModel


class Explainer:
    name = "base"
    model_kinds = ("xgb", "ebm", "lr")

    def explain(self, model: ClinicalModel, x_codes: np.ndarray) -> np.ndarray:
        """x_codes: one patient encoded with FeatureSpace.encode, shape (n_features,)."""
        raise NotImplementedError

    def explain_many(self, model, X_codes):
        return np.vstack([self.explain(model, x) for x in X_codes])


class TreeShapExplainer(Explainer):
    name = "SHAP"
    model_kinds = ("xgb",)

    def explain(self, model, x_codes):
        return self.explain_many(model, np.atleast_2d(x_codes))[0]

    def explain_many(self, model, X_codes):
        # XGBoost's built-in TreeSHAP handles categorical splits natively.
        X = model.space.frame(model.space.decode(X_codes))
        dm = xgb.DMatrix(X, enable_categorical=True)
        contrib = model.estimator.get_booster().predict(dm, pred_contribs=True)
        return contrib[:, :-1]  # last column is the bias term


class EBMTermExplainer(Explainer):
    name = "EBM terms"
    model_kinds = ("ebm",)

    def explain(self, model, x_codes):
        return self.explain_many(model, np.atleast_2d(x_codes))[0]

    def explain_many(self, model, X_codes):
        ebm = model.estimator
        X = model.space.frame(model.space.decode(X_codes))
        scores = ebm.eval_terms(X)  # (n, n_terms) log-odds contributions
        feats = model.space.features
        out = np.zeros((len(X), len(feats)))
        for t, term in enumerate(ebm.term_features_):
            # Pairwise interaction terms are split evenly between their features.
            for f in term:
                out[:, feats.index(ebm.feature_names_in_[f])] += scores[:, t] / len(term)
        return out


class LimeExplainer(Explainer):
    name = "LIME"

    def __init__(self, train_codes: np.ndarray, space, num_samples=config.LIME_SAMPLES, seed=config.SEED):
        self.space = space
        self.num_samples = num_samples
        self.seed = seed
        cat_names = {j: space.levels[c] for j, c in zip(space.categorical_idx, space.categorical)}
        self._lime = LimeTabularExplainer(
            train_codes, feature_names=space.features, class_names=["no", "readmit"],
            categorical_features=space.categorical_idx, categorical_names=cat_names,
            discretize_continuous=True, mode="classification", random_state=seed,
        )

    def explain(self, model, x_codes, seed=None):
        def fn(a):
            p = model.proba_codes(a)
            return np.column_stack([1 - p, p])

        # Reseed every call so the same patient always gets the same explanation.
        # LIME's sampler and discretizer share one RandomState object, so reseed it in place.
        self._lime.random_state.seed(self.seed if seed is None else seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = self._lime.explain_instance(
                np.asarray(x_codes, dtype=float), fn, labels=(1,),
                num_features=len(self.space.features), num_samples=self.num_samples,
            )
        out = np.zeros(len(self.space.features))
        for j, w in exp.local_exp[1]:
            out[j] = w
        return out


class AblationExplainer(Explainer):
    name = "Ablation"

    def __init__(self, background_codes: np.ndarray, n=config.ABLATION_BACKGROUND, seed=config.SEED):
        rng = np.random.default_rng(seed)
        self.bg = background_codes[rng.choice(len(background_codes), n, replace=False)]

    def explain(self, model, x_codes):
        x = np.asarray(x_codes, dtype=float)
        p = len(x)
        base = model.logit_codes(x[None, :])[0]
        # One batch: for each feature j, n copies of x with x_j taken from background.
        batch = np.repeat(x[None, :], p * len(self.bg), axis=0)
        for j in range(p):
            batch[j * len(self.bg):(j + 1) * len(self.bg), j] = self.bg[:, j]
        z = model.logit_codes(batch).reshape(p, len(self.bg))
        return base - z.mean(axis=1)


def explainers_for(model: ClinicalModel, train_codes: np.ndarray):
    """The method set used for a given model."""
    native = {"xgb": TreeShapExplainer(), "ebm": EBMTermExplainer()}.get(model.kind)
    out = [native] if native else []
    out += [LimeExplainer(train_codes, model.space), AblationExplainer(train_codes)]
    return out
