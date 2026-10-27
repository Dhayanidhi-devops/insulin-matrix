from src.data_loader import load_synthetic
from src.database import Database
from src.modeling import save_artifact


def test_dataset_registry_and_experiment_deletion(home):
    database = Database()
    first = load_synthetic(database)
    second = load_synthetic(database)
    assert first["dataset_id"] == second["dataset_id"]
    assert len(database.list_datasets()) == 1
    counts = database.summary_counts()
    assert counts["datasets"] == 1
    assert counts["observations"] == len(first["frame"])
    assert counts["participants"] == 80
    assert counts["models"] == 0

    artifact_id = "md_test"
    relative = f"artifacts/models/{artifact_id}.joblib"
    path = home / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    save_artifact({"app": "Insulin Matrix", "pipeline": {"marker": 1}, "experiment_id": "ex_test"}, path)
    database.insert_artifact(artifact_id, "ex_test", relative)
    database.insert_experiment(
        {
            "experiment_id": "ex_test",
            "dataset_id": first["dataset_id"],
            "dataset_label": first["display_name"],
            "dataset_type": "synthetic",
            "task": "regression",
            "target": "synthetic_metabolic_score",
            "features_json": '["age"]',
            "model_name": "Ridge Regression",
            "hyperparameters_json": '{"alpha": 1.0}',
            "metrics_json": '{"mae": 0.1, "rmse": 0.2, "r2": 0.3}',
            "n_train_participants": 60,
            "n_test_participants": 20,
            "n_train_rows": 70,
            "n_test_rows": 34,
            "seed": 42,
            "test_size": 0.25,
            "artifact_id": artifact_id,
            "split_mode": "participant_grouped",
            "notes": "synthetic",
        }
    )
    assert path.is_file()
    assert database.summary_counts()["latest_experiment"]["experiment_id"] == "ex_test"
    outcome = database.delete_experiment("ex_test")
    assert outcome["experiment_id"] == "ex_test"
    assert not path.exists()
    assert database.get_experiment("ex_test") is None
    assert database.list_experiments() == []

    database.delete_dataset(first["dataset_id"])
    assert database.list_datasets() == []
    assert database.summary_counts()["datasets"] == 0
