"""Counterfactual what-if search: which changes to the actionable features
would lower this patient's predicted risk the most?

Only features marked actionable in the feature dictionary are changed, and
each one only within a plausible range. The output describes what the model's
risk is sensitive to. It is a starting point for discussion, not a treatment
recommendation (treatment recommendations are out of scope, report 4.1).
"""

import numpy as np

from .data import FEATURE_DICTIONARY

# Candidate values per actionable feature. Numeric entries are offsets.
CANDIDATES = {
    "discharge_disposition": ["Home", "Home with health service", "Care facility (SNF/rehab/other)"],
    # Dose adjustments only; the search never suggests stopping a medicine.
    "insulin": ["Steady", "Up", "Down"],
    "time_in_hospital": [-2, -1, +1, +2],
}


def _options(space, x, feature):
    j = space.features.index(feature)
    if feature in space.categorical:
        levels = space.levels[feature]
        return [(j, float(levels.index(v)), v) for v in CANDIDATES[feature]
                if v in levels and levels.index(v) != int(x[j])]
    out = []
    for d in CANDIDATES[feature]:
        v = x[j] + d
        if 1 <= v <= 14:  # length of stay in this data is 1 to 14 days
            out.append((j, float(v), int(v)))
    return out


def search(model, x_codes, max_changes=3, min_gain=0.005):
    """Greedy search. Returns a list of steps: feature, from, to, risk after."""
    space = model.space
    x = np.asarray(x_codes, dtype=float).copy()
    actionable = [f for f, (_, act) in FEATURE_DICTIONARY.items() if act and f in CANDIDATES]
    risk = float(model.proba_codes(x[None, :])[0])
    steps, used = [], set()
    for _ in range(max_changes):
        best = None
        for f in actionable:
            if f in used:
                continue
            for j, code, label in _options(space, x, f):
                trial = x.copy()
                trial[j] = code
                r = float(model.proba_codes(trial[None, :])[0])
                if best is None or r < best[0]:
                    best = (r, f, j, code, label)
        if best is None or risk - best[0] < min_gain:
            break
        r, f, j, code, label = best
        before = space.decode(x[None, :])[f].iloc[0]
        x[j] = code
        steps.append({"feature": f, "from": before, "to": label, "risk_after": r})
        used.add(f)
        risk = r
    return steps
