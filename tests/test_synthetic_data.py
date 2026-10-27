import pandas as pd

from src.synthetic_data import SEED, generate_synthetic_dataset, synthetic_fingerprint


def test_generator_is_reproducible_and_labeled():
    first = generate_synthetic_dataset(seed=SEED)
    second = generate_synthetic_dataset(seed=SEED)
    pd.testing.assert_frame_equal(first, second)
    assert synthetic_fingerprint(first) == synthetic_fingerprint(second)
    other = generate_synthetic_dataset(seed=SEED + 1)
    assert synthetic_fingerprint(first) != synthetic_fingerprint(other)
    assert set(first["data_origin"]) == {"synthetic"}
    assert first["participant_id"].str.startswith("SYN-").all()
    assert first["fasting_glucose_mmol"].min() > 3.5
    assert (first["glucose_30min_mmol"] > first["fasting_glucose_mmol"]).all()
    assert (first["fasting_insulin_uU_ml"] > 0).all()
    assert set(first["synthetic_risk_group"]) == {"higher", "lower"}
    assert first["participant_id"].nunique() == 80
    assert len(first) == 104
