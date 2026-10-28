"""Paths and settings shared by the whole pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

SEED = 42

# Patient-level split fractions (train / calibration / test).
SPLIT = {"train": 0.70, "calib": 0.15, "test": 0.15}

# Explanation and reliability settings.
TOP_K = 5                 # features compared by the overlap measure
LIME_SAMPLES = 1000       # perturbations per LIME explanation
ABLATION_BACKGROUND = 50  # background rows for the ablation explainer
N_STABILITY_MODELS = 5    # models retrained on resampled patients
RELIABILITY_WEIGHTS = {"agreement": 1 / 3, "stability": 1 / 3, "overlap": 1 / 3}

# Reliability bands (High / Moderate / Low), set from the score distribution on 2,000 test
# patients (results/rq1_rq3_xgb.md): Low is about the bottom 17%, High the top 30%.
RELIABILITY_BANDS = {"high": 0.75, "moderate": 0.65}


def ensure_dirs():
    for p in (ARTIFACTS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
