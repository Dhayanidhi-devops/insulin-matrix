import math

import numpy as np
import pandas as pd
import pytest

from src.errors import UnitError
from src.indices import (
    HOMA2_STATEMENT,
    INDEX_COLUMNS,
    calculate_indices,
    homa_b,
    homa_ir,
    insulinogenic_index,
    matsuda_index,
    oral_disposition_index,
    trapezoidal_auc,
)
from src.synthetic_data import generate_synthetic_dataset


def test_homa_ir_and_homa_b_agree_across_units():
    mmol = homa_ir(5.0, 10.0, "mmol/L", "µU/mL")
    mgdl = homa_ir(90.0, 10.0, "mg/dL", "uU/mL")
    pmol = homa_ir(5.0, 69.45, "mmol/L", "pmol/L")
    assert mmol.value == pytest.approx(50 / 22.5)
    assert mgdl.value == pytest.approx(mmol.value)
    assert pmol.value == pytest.approx(mmol.value)

    beta_mmol = homa_b(5.0, 10.0, "mmol/L", "µU/mL")
    beta_mg = homa_b(90.0, 10.0, "mg/dL", "mIU/L")
    assert beta_mmol.value == pytest.approx(200 / 1.5)
    assert beta_mg.value == pytest.approx(beta_mmol.value)
    assert homa_b(3.5, 10, "mmol/L", "µU/mL").value is None
    assert homa_b(63, 10, "mg/dL", "µU/mL").value is None
    assert "non-positive" in homa_b(3.4, 10, "mmol/L", "µU/mL").message


def test_homa_rejects_invalid_numbers_and_units():
    assert homa_ir(-1, 10, "mmol/L", "µU/mL").value is None
    assert homa_ir(5, None, "mmol/L", "µU/mL").value is None
    with pytest.raises(UnitError):
        homa_ir(5, 10, "g/L", "µU/mL")


def test_insulinogenic_index_and_disposition():
    igi = insulinogenic_index(5, 8, 10, 40, "mmol/L", "µU/mL")
    same = insulinogenic_index(90, 144, 10, 40, "mg/dL", "µU/mL")
    assert igi.value == pytest.approx(10)
    assert same.value == pytest.approx(10)
    assert insulinogenic_index(8, 5, 10, 40, "mmol/L", "µU/mL").value is None
    assert insulinogenic_index(5, 5, 10, 40, "mmol/L", "µU/mL").message.startswith("Insulinogenic")
    odi = oral_disposition_index(igi, 10, "µU/mL")
    assert odi.value == pytest.approx(1)
    assert oral_disposition_index(igi, 0, "µU/mL").value is None


def test_trapezoidal_auc_and_time_rules():
    flat = trapezoidal_auc([0, 60], [5, 5])
    assert flat.value == pytest.approx(300)
    assert flat.status == "warning"
    three = trapezoidal_auc([0, 30, 60], [5, 7, 5])
    assert three.value == pytest.approx(360)
    assert trapezoidal_auc([0], [5]).value is None
    assert trapezoidal_auc([30, 0], [5, 6]).value is None
    assert trapezoidal_auc([0, 30], [5, -1]).value is None
    full = trapezoidal_auc([0, 30, 60, 90, 120], [5, 6, 7, 6, 5])
    assert full.status == "ok"
    assert full.value == pytest.approx(
        30 * (5 + 6) / 2 + 30 * (6 + 7) / 2 + 30 * (7 + 6) / 2 + 30 * (6 + 5) / 2
    )


def test_matsuda_matches_hand_calculation_and_rejects_zero_insulin():
    glucose = {0: 90, 30: 140, 60: 130, 120: 110}
    insulin = {0: 10, 30: 50, 60: 40, 120: 30}
    result = matsuda_index(glucose, insulin, "mg/dL", "µU/mL")
    mean_g = np.mean(list(glucose.values()))
    mean_i = np.mean(list(insulin.values()))
    expected = 10000 / math.sqrt(90 * 10 * mean_g * mean_i)
    assert result.value == pytest.approx(expected)
    assert result.status == "warning"
    zero = matsuda_index({0: 90, 30: 140}, {0: 0, 30: 50}, "mg/dL", "µU/mL")
    assert zero.value is None


def test_homa2_is_not_calculated_and_synthetic_rows_are_complete():
    frame = generate_synthetic_dataset().rename(
        columns={
            "fasting_glucose_mmol": "fasting_glucose",
            "fasting_insulin_uU_ml": "fasting_insulin",
            "glucose_30min_mmol": "glucose_30",
            "glucose_60min_mmol": "glucose_60",
            "glucose_90min_mmol": "glucose_90",
            "glucose_120min_mmol": "glucose_120",
            "insulin_30min_uU_ml": "insulin_30",
            "insulin_60min_uU_ml": "insulin_60",
            "insulin_90min_uU_ml": "insulin_90",
            "insulin_120min_uU_ml": "insulin_120",
        }
    )
    result, summary = calculate_indices(frame, {"glucose": "mmol/L", "insulin": "µU/mL"})
    assert "homa2" not in result.columns
    assert HOMA2_STATEMENT in " ".join(summary["homa2"].astype(str))
    for column in INDEX_COLUMNS:
        assert result[column].notna().all()
    manual = frame["fasting_glucose"] * frame["fasting_insulin"] / 22.5
    assert result["homa_ir"].to_numpy() == pytest.approx(manual.to_numpy())


def test_calculate_indices_leaves_failed_rows_blank():
    frame = pd.DataFrame(
        {
            "fasting_glucose": [5.0, 3.0],
            "fasting_insulin": [10.0, 10.0],
            "glucose_30": [8.0, 2.0],
            "insulin_30": [40.0, 12.0],
        }
    )
    result, _summary = calculate_indices(frame, {"glucose": "mmol/L", "insulin": "µU/mL"})
    assert result.loc[0, "homa_b"] == pytest.approx(200 / 1.5)
    assert pd.isna(result.loc[1, "homa_b"])
    assert "HOMA-B" in result.loc[1, "calculation_notes"]
    assert pd.isna(result.loc[1, "insulinogenic_index"])
