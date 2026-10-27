from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import EXCLUDED_FROM_FEATURES
from src.modeling import TrainConfig, available_models
from src.services import attach_column_units, prepare_training_frame, remember_frame, train_and_record
from src.synthetic_data import SYNTHETIC_FEATURES
from src.ui import RESULT_KEY, page_header, require_validated, show_exception, sync_dataset_widgets

page_header(
    "Model Training",
    "Fit one model and score participants who were not used to train it. Review the features and preprocessing before you start.",
)
active = require_validated()
sync_dataset_widgets(active)
frame = active["analysis_frame"]
candidates = [column for column in frame.columns if column not in EXCLUDED_FROM_FEATURES]
if not candidates:
    st.error("The analysis table has no usable columns.")
    st.stop()

if active["dataset_type"] == "synthetic":
    st.caption("These setups predict the documented synthetic target. They are not clinical models.")
    left, right = st.columns(2)
    if left.button("Use synthetic regression setup"):
        st.session_state["feature_selection"] = [column for column in SYNTHETIC_FEATURES if column in candidates]
        st.session_state["model_target"] = "synthetic_metabolic_score"
        st.session_state["model_task"] = "Regression"
        st.session_state["model_name"] = "Random Forest Regressor"
        st.rerun()
    if right.button("Use synthetic classification setup"):
        st.session_state["feature_selection"] = [column for column in SYNTHETIC_FEATURES if column in candidates]
        st.session_state["model_target"] = "synthetic_risk_group"
        st.session_state["model_task"] = "Classification"
        st.session_state["model_name"] = "Random Forest Classifier"
        st.rerun()

if "feature_selection" in st.session_state:
    st.session_state["feature_selection"] = [column for column in st.session_state["feature_selection"] if column in candidates]
features = st.multiselect("Features", candidates, key="feature_selection")
if "model_target" not in st.session_state:
    preferred = active.get("target_column")
    st.session_state["model_target"] = preferred if preferred in candidates else candidates[0]
if st.session_state.get("model_target") not in candidates:
    st.session_state["model_target"] = candidates[0]
target = st.selectbox("Target", candidates, key="model_target")
task_label = st.radio("Task", ["Regression", "Classification"], key="model_task")
task = "regression" if task_label == "Regression" else "classification"

models = available_models(task)
model_names = [model["name"] for model in models]
if "model_name" not in st.session_state or st.session_state["model_name"] not in model_names:
    st.session_state["model_name"] = model_names[0]
model_name = st.selectbox("Model", model_names, key="model_name")
spec = next(model for model in models if model["name"] == model_name)
hyperparameters = {}
for param in spec["params"]:
    label = param["key"].replace("_", " ")
    help_text = "Use 0 for unlimited depth." if param["key"] == "max_depth" and param["min"] == 0 else None
    hyperparameters[param["key"]] = st.number_input(
        label,
        min_value=param["min"],
        max_value=param["max"],
        value=param["default"],
        step=param["step"],
        key=f"hp_{model_name}_{param['key']}",
        help=help_text,
    )
if not spec["params"]:
    st.caption("This model has no extra hyperparameters. Linear regression uses ordinary least squares.")

seed = int(st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1, key="train_seed"))
test_size = float(st.slider("Test proportion", min_value=0.10, max_value=0.40, value=0.25, step=0.05, key="train_test_size"))
scale_numeric = st.checkbox("Standardize numeric features", value=True, key="scale_numeric")
balanced = False
positive_class = None
if task == "classification":
    balanced = st.checkbox("Balance class weights", key="class_weight_balanced")
    if target in frame.columns:
        classes = sorted(frame[target].dropna().astype(str).unique().tolist())
        if len(classes) == 2:
            positive_class = st.selectbox("Positive class for precision, recall, F1, and ROC-AUC", classes, key="positive_class")

has_participant = "participant_id" in frame.columns
allow_rows = False
if not has_participant:
    st.warning("No participant ID is mapped. Row splitting stays off unless you confirm that each row is an independent participant.")
    allow_rows = st.checkbox("Each row is an independent participant", key="allow_row_split")

derived = list(st.session_state.get("derived_features") or [])
normalize = bool(st.session_state.get("normalize_units", False))
missing_strategy = st.session_state.get("missing_strategy", "pipeline_impute")
numeric_impute = st.session_state.get("numeric_impute", "median")
drop_duplicates = bool(st.session_state.get("drop_exact_duplicates", False))

st.subheader("Configuration to be trained")
st.dataframe(
    [
        {"setting": "Features", "value": ", ".join(features) or "None"},
        {"setting": "Target", "value": target},
        {"setting": "Task", "value": task},
        {"setting": "Model", "value": model_name},
        {"setting": "Hyperparameters", "value": str(hyperparameters)},
        {"setting": "Seed", "value": str(seed)},
        {"setting": "Test proportion", "value": str(test_size)},
        {"setting": "Missing values", "value": str(missing_strategy)},
        {"setting": "Numeric impute", "value": str(numeric_impute)},
        {"setting": "Standardize numerics", "value": str(bool(scale_numeric))},
        {"setting": "Derived columns", "value": ", ".join(derived) or "None"},
        {"setting": "Unit normalization", "value": str(bool(normalize))},
        {"setting": "Drop exact duplicates", "value": str(bool(drop_duplicates))},
        {"setting": "Split", "value": "participant holdout" if has_participant else ("acknowledged row split" if allow_rows else "blocked")},
    ],
    width="stretch",
    hide_index=True,
)
st.caption("Test metrics are computed after the fit. Training-set metrics are stored separately and labeled with a train_ prefix.")

if st.button("Train model", type="primary"):
    try:
        model_frame, prep_messages, extra_units = prepare_training_frame(active, derived, normalize)
        config = TrainConfig(
            task=task,
            model_name=model_name,
            target=target,
            features=list(features),
            seed=seed,
            test_size=test_size,
            hyperparameters=hyperparameters,
            scale_numeric=bool(scale_numeric),
            numeric_impute=numeric_impute if numeric_impute in {"median", "mean"} else "median",
            missing_strategy=missing_strategy if missing_strategy in {"pipeline_impute", "drop_incomplete"} else "pipeline_impute",
            drop_exact_duplicates=drop_duplicates,
            group_column="participant_id" if has_participant else None,
            allow_row_split=allow_rows,
            positive_class=positive_class,
            class_weight_balanced=balanced,
            dataset_type=active["dataset_type"],
            units=active.get("units") or {},
        )
        attach_column_units(config, list(features), active.get("units") or {}, extra_units)
        remember_frame(config, model_frame)
        with st.spinner("Training on the participant holdout..."):
            result, experiment_id = train_and_record(active, config, prep_messages)
        st.session_state[RESULT_KEY] = {
            "experiment_id": experiment_id,
            "metrics": result.metrics,
            "messages": result.messages,
            "evaluation": result.evaluation,
            "importance": None if result.importance is None else result.importance,
            "permutation": None if result.permutation is None else result.permutation,
            "model_name": result.model_name,
            "target": result.target,
            "task": result.task,
            "features": result.features,
            "dataset_type": result.dataset_type,
            "dataset_id": active["dataset_id"],
            "split_mode": result.split_mode,
            "preprocessing": result.preprocessing,
            "hyperparameters": result.hyperparameters,
            "seed": result.seed,
            "test_size": result.test_size,
            "n_train_participants": result.n_train_participants,
            "n_test_participants": result.n_test_participants,
        }
        st.success(f"Saved experiment {experiment_id}. Open Model Evaluation for plots and the comparison table.")
        st.json({key: value for key, value in result.metrics.items() if not str(key).startswith("train_")})
        for message in result.messages:
            st.warning(message)
    except Exception as exc:
        show_exception(exc)
