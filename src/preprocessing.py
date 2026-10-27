"""Train-only preprocessing and participant-level splitting.

Imputation and scaling statistics are learned inside a scikit-learn pipeline
that is fitted on the training partition only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.errors import ModelingError


def split_feature_types(frame: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for feature in features:
        series = frame[feature]
        if pd.api.types.is_bool_dtype(series):
            categorical.append(feature)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(feature)
        else:
            parsed = pd.to_numeric(series, errors="coerce")
            original = series.notna()
            if int(original.sum()) and int((original & parsed.notna()).sum()) == int(original.sum()):
                numeric.append(feature)
            else:
                categorical.append(feature)
    return numeric, categorical


def coerce_model_frame(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Parse numeric-looking feature columns. Raises if any value cannot be parsed."""
    messages: list[str] = []
    out = frame.copy()
    for feature in features:
        series = out[feature]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            continue
        parsed = pd.to_numeric(series, errors="coerce")
        original = series.notna()
        failed = int((original & parsed.isna()).sum())
        if failed == 0 and int(original.sum()):
            out[feature] = parsed
            messages.append(
                f"Parsed text values in '{feature}' as numbers in the modelling copy. The uploaded file was not modified."
            )
        elif failed and int((original & parsed.notna()).sum()) == 0:
            continue
        elif failed:
            raise ModelingError(
                f"{failed} values in '{feature}' could not be parsed as numbers. "
                "Fix the column or remove it from the feature list."
            )
    return out, messages


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str], *, scale_numeric: bool, numeric_impute: str) -> ColumnTransformer:
    if numeric_impute not in {"median", "mean"}:
        raise ModelingError("Numeric imputation must be median or mean.")
    transformers = []
    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy=numeric_impute))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_columns))
    if not transformers:
        raise ModelingError("Select at least one feature.")
    return ColumnTransformer(transformers, remainder="drop")


def rows_with_missing_target(frame: pd.DataFrame, target: str) -> int:
    return int(frame[target].isna().sum())


def incomplete_feature_rows(frame: pd.DataFrame, features: list[str]) -> int:
    if not features or frame.empty:
        return 0
    return int(frame.loc[:, features].isna().any(axis=1).sum())


def constant_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    found = []
    for feature in features:
        values = frame[feature].dropna()
        if values.empty or values.nunique(dropna=True) <= 1:
            found.append(feature)
    return found


def _participant_labels(frame: pd.DataFrame, group_column: str, stratify: pd.Series | None):
    groups = frame[group_column].astype(str).str.strip()
    meta = pd.DataFrame({"group": groups.drop_duplicates().to_numpy()})
    if stratify is None:
        meta["label"] = ""
        return meta, groups, []
    labels = stratify.astype(str)
    paired = pd.DataFrame({"group": groups.to_numpy(), "label": labels.to_numpy()})
    messages = []
    modes = []
    mixed = 0
    for group, part in paired.groupby("group", sort=False):
        counts = part["label"].value_counts(sort=False)
        top = counts.max()
        choice = sorted(label for label, count in counts.items() if count == top)[0]
        if len(counts) > 1:
            mixed += 1
        modes.append((group, choice))
    if mixed:
        messages.append(
            f"{mixed} participants have more than one target label. "
            "Stratification used the most common label for each participant, with a sorted-name tie break."
        )
    return pd.DataFrame(modes, columns=["group", "label"]), groups, messages


def split_participants(
    frame: pd.DataFrame,
    group_column: str,
    test_size: float,
    seed: int,
    stratify_labels: pd.Series | None = None,
) -> tuple[pd.Index, pd.Index, dict]:
    """Hold out whole participants. Every row of a participant stays in one partition."""
    if group_column not in frame.columns:
        raise ModelingError("A participant identifier is required for grouped splitting.")
    if not 0.1 <= float(test_size) <= 0.4:
        raise ModelingError("The test proportion must be between 0.1 and 0.4.")
    if frame[group_column].isna().any():
        raise ModelingError("Participant identifiers contain missing values, so a grouped split cannot be made.")
    meta, aligned_groups, messages = _participant_labels(frame, group_column, stratify_labels)
    if aligned_groups.eq("").any():
        raise ModelingError("Participant identifiers contain missing values, so a grouped split cannot be made.")
    unique_groups = aligned_groups.drop_duplicates().tolist()
    n_groups = len(unique_groups)
    if n_groups < 4:
        raise ModelingError(
            f"Only {n_groups} participants are available. At least 4 are required to create a training partition and a test partition."
        )
    n_test = int(round(n_groups * float(test_size)))
    n_test = min(max(n_test, 2), n_groups - 2)
    if n_test != int(round(n_groups * float(test_size))):
        messages.append(
            f"The test partition was adjusted to {n_test} participants so both partitions contain at least 2 participants."
        )
    if stratify_labels is None:
        order = np.array(unique_groups)
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        test_groups = set(order[:n_test].tolist())
    else:
        class_counts = meta["label"].value_counts()
        if int(class_counts.min()) < 2:
            raise ModelingError(
                "At least one class is present in fewer than 2 participants. "
                "A stratified participant holdout cannot place that class in both partitions, "
                "so a reliable classification evaluation cannot be performed."
            )
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
            positions = np.arange(len(meta))
            _, test_positions = next(splitter.split(positions, meta["label"]))
        except ValueError as exc:
            raise ModelingError(
                "A stratified participant holdout could not be constructed. "
                "A class may have too few participants for this test proportion. "
                f"{exc}"
            ) from exc
        test_groups = set(meta.iloc[test_positions]["group"].astype(str).tolist())

    test_mask = aligned_groups.isin(test_groups).to_numpy()
    train_index = frame.index[~test_mask]
    test_index = frame.index[test_mask]
    train_groups = set(aligned_groups.loc[train_index].tolist())
    if train_groups & test_groups:
        raise ModelingError("Internal error: a participant was placed in both partitions.")
    if aligned_groups.loc[train_index].isin(test_groups).any():
        raise ModelingError("Internal error: a participant row crossed the holdout boundary.")
    info = {
        "split_mode": "participant_grouped",
        "n_train_participants": len(train_groups),
        "n_test_participants": len(test_groups),
        "n_train_rows": int(len(train_index)),
        "n_test_rows": int(len(test_index)),
        "messages": messages,
    }
    return train_index, test_index, info


def split_independent_rows(frame: pd.DataFrame, test_size: float, seed: int, stratify_labels: pd.Series | None = None):
    """Split rows only after the user confirms that each row is one participant."""
    working = frame.copy()
    working["_row_participant"] = [f"row-{position}" for position in range(len(working))]
    labels = None if stratify_labels is None else pd.Series(stratify_labels.to_numpy(), index=working.index)
    train_index, test_index, info = split_participants(
        working,
        "_row_participant",
        test_size,
        seed,
        labels,
    )
    info["split_mode"] = "row_independent_acknowledged"
    info["messages"] = [
        "Rows were treated as independent participants because no participant identifier is mapped "
        "and the independent-row confirmation was set."
    ] + info["messages"]
    return train_index, test_index, info
