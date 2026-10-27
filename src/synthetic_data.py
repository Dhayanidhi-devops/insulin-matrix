"""Deterministic synthetic demonstration data for Insulin Matrix.

The generator does not describe real people. The modelling target is a
documented function of the simulated fasting values plus noise. Performance
against that target is not clinical evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_PARTICIPANTS = 80
N_REPEAT_VISITS = 24

SYNTHETIC_DATASET_NAME = "Synthetic demonstration dataset"
TARGET_DEFINITION = (
    "synthetic_metabolic_score = 1.4 × HOMA1-IR + 0.015 × age_years "
    "+ 0.02 × (bmi − 25) + noise. Noise is drawn from a normal distribution "
    "with mean 0 and standard deviation 0.15, using the dataset seed. "
    "HOMA1-IR uses that row's fasting glucose in mmol/L and fasting insulin "
    "in µU/mL. synthetic_risk_group is a median split of the score into "
    "higher and lower. Neither column is a disease label."
)
GENERATION_NOTES = (
    "Eighty synthetic participants are created with NumPy Generator seed 42. "
    "The first 24 participants also have a second visit whose glucose and "
    "insulin values are small perturbations of the first visit. "
    "Age, sex, and BMI stay the same across visits. "
    "Every fasting glucose is above 3.5 mmol/L, every 30-minute glucose is "
    "higher than the fasting glucose, and insulin values are positive, so "
    "the documented indices can be calculated. "
    + TARGET_DEFINITION
)

SYNTHETIC_MAPPING = {
    "participant_id": "participant_id",
    "observation_id": "visit_id",
    "age": "age_years",
    "sex": "sex",
    "bmi": "bmi",
    "waist_cm": "waist_cm",
    "fasting_glucose": "fasting_glucose_mmol",
    "fasting_insulin": "fasting_insulin_uU_ml",
    "glucose_30": "glucose_30min_mmol",
    "glucose_60": "glucose_60min_mmol",
    "glucose_90": "glucose_90min_mmol",
    "glucose_120": "glucose_120min_mmol",
    "insulin_30": "insulin_30min_uU_ml",
    "insulin_60": "insulin_60min_uU_ml",
    "insulin_90": "insulin_90min_uU_ml",
    "insulin_120": "insulin_120min_uU_ml",
    "target": "synthetic_metabolic_score",
}
SYNTHETIC_EXTRA_COVARIATES: list[str] = []
SYNTHETIC_UNITS = {"glucose": "mmol/L", "insulin": "µU/mL"}
SYNTHETIC_FEATURES = ["age", "sex", "bmi", "waist_cm", "fasting_glucose", "fasting_insulin"]


def _homa(glucose_mmol: np.ndarray, insulin_uU: np.ndarray) -> np.ndarray:
    return glucose_mmol * insulin_uU / 22.5


def _clip_positive(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _base_table(rng: np.random.Generator, n: int) -> pd.DataFrame:
    latent = rng.normal(0.0, 1.0, n)
    age = rng.integers(25, 76, n)
    sex = rng.choice(np.array(["Female", "Male"]), size=n)
    bmi = _clip_positive(26.5 + 3.2 * latent + rng.normal(0.0, 1.1, n), 18.5, 42.0)
    waist = _clip_positive(78 + 2.1 * (bmi - 25) + rng.normal(0.0, 2.0, n), 62.0, 130.0)
    fasting_glucose = _clip_positive(5.15 + 0.42 * latent + rng.normal(0.0, 0.12, n), 4.4, 8.8)
    fasting_insulin = _clip_positive(7.5 + 3.4 * latent + rng.normal(0.0, 0.7, n), 2.5, 38.0)
    glucose_30 = fasting_glucose + _clip_positive(2.1 + 0.25 * latent + rng.normal(0.0, 0.15, n), 0.8, 5.5)
    glucose_60 = fasting_glucose + _clip_positive(2.4 + 0.55 * latent + rng.normal(0.0, 0.2, n), 0.6, 7.0)
    glucose_90 = fasting_glucose + _clip_positive(1.5 + 0.45 * latent + rng.normal(0.0, 0.18, n), 0.3, 6.0)
    glucose_120 = fasting_glucose + _clip_positive(0.7 + 0.35 * latent + rng.normal(0.0, 0.16, n), 0.1, 5.0)
    insulin_30 = fasting_insulin + _clip_positive(22 + 7 * latent + rng.normal(0.0, 1.5, n), 8.0, 90.0)
    insulin_60 = fasting_insulin + _clip_positive(28 + 9 * latent + rng.normal(0.0, 1.8, n), 8.0, 110.0)
    insulin_90 = fasting_insulin + _clip_positive(16 + 6 * latent + rng.normal(0.0, 1.4, n), 5.0, 80.0)
    insulin_120 = fasting_insulin + _clip_positive(8 + 3 * latent + rng.normal(0.0, 1.0, n), 3.0, 50.0)
    return pd.DataFrame(
        {
            "participant_id": [f"SYN-{index + 1:03d}" for index in range(n)],
            "visit_id": "V1",
            "age_years": age.astype(int),
            "sex": sex,
            "bmi": bmi,
            "waist_cm": waist,
            "fasting_glucose_mmol": fasting_glucose,
            "fasting_insulin_uU_ml": fasting_insulin,
            "glucose_30min_mmol": glucose_30,
            "glucose_60min_mmol": glucose_60,
            "glucose_90min_mmol": glucose_90,
            "glucose_120min_mmol": glucose_120,
            "insulin_30min_uU_ml": insulin_30,
            "insulin_60min_uU_ml": insulin_60,
            "insulin_90min_uU_ml": insulin_90,
            "insulin_120min_uU_ml": insulin_120,
        }
    )


def _second_visit(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    second = base.iloc[:N_REPEAT_VISITS].copy()
    second["visit_id"] = "V2"
    glucose_cols = [col for col in second.columns if "glucose" in col]
    insulin_cols = [col for col in second.columns if "insulin" in col]
    for column in glucose_cols:
        factor = rng.uniform(0.97, 1.03, len(second))
        second[column] = second[column].to_numpy() * factor
    for column in insulin_cols:
        factor = rng.uniform(0.96, 1.04, len(second))
        second[column] = second[column].to_numpy() * factor
    second["fasting_glucose_mmol"] = np.clip(second["fasting_glucose_mmol"], 4.4, 8.8)
    gap = second["glucose_30min_mmol"] - second["fasting_glucose_mmol"]
    second.loc[gap < 0.8, "glucose_30min_mmol"] = second.loc[gap < 0.8, "fasting_glucose_mmol"] + 0.8
    for column in insulin_cols:
        second[column] = np.clip(second[column], 2.5, None)
    return second


def _attach_targets(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    homa = _homa(out["fasting_glucose_mmol"].to_numpy(), out["fasting_insulin_uU_ml"].to_numpy())
    noise = rng.normal(0.0, 0.15, len(out))
    score = 1.4 * homa + 0.015 * out["age_years"].to_numpy() + 0.02 * (out["bmi"].to_numpy() - 25.0) + noise
    out["synthetic_metabolic_score"] = score
    median = float(np.median(score))
    out["synthetic_risk_group"] = np.where(score >= median, "higher", "lower")
    out["data_origin"] = "synthetic"
    out["generator_seed"] = SEED
    return out


def generate_synthetic_dataset(
    n_participants: int = N_PARTICIPANTS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a wide synthetic OGTT table. The same seed always returns the same rows."""
    if n_participants < N_REPEAT_VISITS:
        raise ValueError("The demonstration generator needs at least 24 participants.")
    rng = np.random.default_rng(seed)
    base = _base_table(rng, n_participants)
    second = _second_visit(base, rng)
    combined = pd.concat([base, second], ignore_index=True)
    combined = combined.sort_values(["participant_id", "visit_id"], kind="mergesort").reset_index(drop=True)
    # The target noise stream is drawn after the table is ordered, so the seed
    # fully determines both visits and the median split.
    target_rng = np.random.default_rng(seed + 10_000)
    return _attach_targets(combined, target_rng)


def synthetic_fingerprint(frame: pd.DataFrame) -> str:
    """Stable checksum of rounded numeric columns plus identifiers."""
    import hashlib

    numeric = frame.select_dtypes(include="number").round(8)
    payload = frame.assign(**{column: numeric[column] for column in numeric.columns})
    text = payload.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
