import pandas as pd

from src.synthetic_data import SYNTHETIC_MAPPING, SYNTHETIC_UNITS, generate_synthetic_dataset
from src.validation import empty_mapping, validate_dataset


def _wide_frame():
    return pd.DataFrame(
        {
            "pid": ["A", "B", "A"],
            "visit": ["V1", "V1", "V1"],
            "glucose": [5.0, 6.0, 5.5],
            "insulin": [10.0, 12.0, 9.0],
            "glucose30": [8.0, 9.0, 8.5],
            "insulin30": [40.0, 30.0, 28.0],
            "age": [40, 50, 40],
        }
    )


def test_synthetic_mapping_passes():
    frame = generate_synthetic_dataset()
    _analysis, report = validate_dataset(frame, SYNTHETIC_MAPPING, SYNTHETIC_UNITS, content_hash="synthetic")
    assert report.status == "passed"
    assert report.layout == "wide"
    assert report.target_column == "synthetic_metabolic_score"
    assert report.readiness["homa_ir"] == "ready"
    assert report.readiness["matsuda_index"] == "ready"


def test_missing_column_and_unsupported_unit():
    frame = _wide_frame()
    mapping = empty_mapping()
    mapping["fasting_glucose"] = "does_not_exist"
    mapping["fasting_insulin"] = "insulin"
    _analysis, report = validate_dataset(frame, mapping, {"glucose": "mmol/L", "insulin": "µU/mL"})
    assert report.has_errors()
    assert "MISSING_COLUMN" in set(report.to_frame()["code"])

    mapping = empty_mapping()
    mapping["fasting_glucose"] = "glucose"
    mapping["fasting_insulin"] = "insulin"
    _analysis, report = validate_dataset(frame, mapping, {"glucose": "g/L", "insulin": "µU/mL"})
    assert "UNSUPPORTED_UNIT" in set(report.to_frame()["code"])
    errors = report.to_frame()
    assert set(errors.loc[errors["code"] == "UNSUPPORTED_UNIT", "severity"]) == {"error"}


def test_duplicate_keys_are_errors_and_missing_values_are_warnings():
    frame = _wide_frame()
    mapping = empty_mapping()
    mapping.update(
        {
            "participant_id": "pid",
            "observation_id": "visit",
            "fasting_glucose": "glucose",
            "fasting_insulin": "insulin",
            "age": "age",
        }
    )
    _analysis, report = validate_dataset(frame, mapping, {"glucose": "mmol/L", "insulin": "µU/mL"})
    codes = set(report.to_frame()["code"])
    assert "DUPLICATE_KEYS" in codes
    assert report.has_errors()

    frame.loc[0, "glucose"] = None
    frame = frame.drop_duplicates(["pid", "visit"])
    _analysis, report = validate_dataset(frame, mapping, {"glucose": "mmol/L", "insulin": "µU/mL"})
    warnings = report.to_frame()
    assert "MISSING_VALUES" in set(warnings["code"])
    assert "error" not in set(warnings.loc[warnings["code"] == "MISSING_VALUES", "severity"])


def test_impossible_glucose_is_an_error():
    frame = pd.DataFrame({"pid": ["A"], "glucose": [-4], "insulin": [8]})
    mapping = empty_mapping()
    mapping.update({"participant_id": "pid", "fasting_glucose": "glucose", "fasting_insulin": "insulin"})
    _analysis, report = validate_dataset(frame, mapping, SYNTHETIC_UNITS)
    assert "IMPOSSIBLE_VALUE" in set(report.to_frame()["code"])
    assert report.has_errors()


def test_long_format_pivots_time_points():
    frame = pd.DataFrame(
        {
            "pid": ["A", "A", "A"],
            "visit": ["V1", "V1", "V1"],
            "minute": [0, 30, 120],
            "gluc": [5.0, 8.0, 6.0],
            "ins": [10.0, 40.0, 20.0],
            "age": [40, 40, 40],
        }
    )
    mapping = empty_mapping()
    mapping.update(
        {
            "participant_id": "pid",
            "observation_id": "visit",
            "time_point": "minute",
            "glucose": "gluc",
            "insulin": "ins",
            "age": "age",
            "target": "age",
        }
    )
    analysis, report = validate_dataset(frame, mapping, SYNTHETIC_UNITS)
    assert not report.has_errors(), report.to_frame()
    assert report.layout == "long"
    assert len(analysis) == 1
    assert analysis.loc[0, "fasting_glucose"] == 5
    assert analysis.loc[0, "glucose_30"] == 8
    assert analysis.loc[0, "glucose_120"] == 6
    assert analysis.loc[0, "fasting_insulin"] == 10
    assert "INADEQUATE_TIMEPOINTS" in set(report.to_frame()["code"])


def test_target_leakage_helper():
    from src.validation import leakage_issues

    issues = leakage_issues(["age", "homa_ir"], "homa_ir")
    assert any(issue.severity == "error" and issue.code == "TARGET_LEAKAGE" for issue in issues)
    issues = leakage_issues(["participant_id"], "age")
    assert any(issue.code == "TARGET_LEAKAGE" for issue in issues)
    issues = leakage_issues(["fasting_glucose", "fasting_insulin"], "homa_ir")
    assert any(issue.code == "POTENTIAL_LEAKAGE" and issue.severity == "warning" for issue in issues)
