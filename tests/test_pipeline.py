"""Run with:  pytest -q"""

import numpy as np
import pandas as pd
import pytest

from xaicdss import data, reliability


@pytest.fixture(scope="module")
def df():
    return data.build_features(data.load_raw())


def test_no_patient_in_two_splits(df):
    splits = data.patient_split(df)
    assert data.check_no_leakage(splits)
    pats = [set(s["patient_nbr"]) for s in splits.values()]
    assert not (pats[0] & pats[1]) and not (pats[0] & pats[2]) and not (pats[1] & pats[2])
    assert sum(len(s) for s in splits.values()) == len(df)


def test_leakage_check_catches_a_shared_patient(df):
    splits = data.patient_split(df)
    bad = dict(splits)
    bad["test"] = pd.concat([splits["test"], splits["train"].head(1)])
    with pytest.raises(data.LeakageError):
        data.check_no_leakage(bad)


def test_ids_and_target_are_not_features():
    with pytest.raises(data.LeakageError):
        data.check_no_leakage({"a": pd.DataFrame({"patient_nbr": [], "encounter_id": []})},
                              feature_names=data.FEATURES + ["patient_nbr"])


def test_weight_not_used_and_target_rate(df):
    assert "weight" not in data.FEATURES
    assert 0.10 < df[data.TARGET].mean() < 0.13


def test_encode_decode_roundtrip(df):
    space = data.FeatureSpace.fit(df)
    sample = df.head(50)
    back = space.decode(space.encode(sample))
    pd.testing.assert_frame_equal(back.astype(str), space.frame(sample).reset_index(drop=True).astype(str))


def test_reliability_measures():
    a = np.array([5, 4, 3, 2, 1, 0, 0, 0.0])
    assert reliability.rank_agreement(a, a) == pytest.approx(1.0)
    assert reliability.rank_agreement(a, -a) == pytest.approx(0.0)
    assert reliability.jaccard(a, a) == 1.0
    score = reliability.combine(1.0, 1.0, 1.0)
    assert score == pytest.approx(1.0)
    assert reliability.band(0.9) == "High" and reliability.band(0.1) == "Low"
