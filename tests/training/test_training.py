import pytest
import json
import pandas as pd
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
import logging
import sys
from unittest.mock import patch

# Import the main app
script_folder = Path(__file__).parents[2] / "src" / "training"
sys.path.append(str(script_folder))
from train import main  # Replace with the actual name of your training script

# Paths
data_folder = script_folder.parents[1] / "data"
training_data_path = data_folder / "training" / "training_data.parquet"
model_file = script_folder.parent / "model_package" / "expected_runs_model.pkl"

test_result_folder = data_folder / "tests"
os.makedirs(test_result_folder, exist_ok=True)
log_file = test_result_folder / "test_training.json"


@pytest.fixture
def training_data():
    """Load the training data for tests."""
    assert training_data_path.exists(), "Training data file is missing"
    df = pd.read_parquet(training_data_path)
    assert not df.empty, "Training data should not be empty"
    return df


def test_train_test_split(training_data):
    """Test the group-based train-test split."""
    input_features = [
        "initial_batter",
        "initial_bowler",
        "num_batsmen",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
    ]
    target = "runs"
    groups = training_data["matchid"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        gss.split(training_data[input_features], training_data[target], groups=groups)
    )

    train_ids = set(training_data.loc[train_idx, "matchid"])
    test_ids = set(training_data.loc[test_idx, "matchid"])

    assert train_ids.isdisjoint(test_ids), "Train-test split has overlapping match IDs"
    assert len(train_ids) > 0, "Train set should not be empty"
    assert len(test_ids) > 0, "Test set should not be empty"


def test_model_training(training_data):
    """Test if the model trains successfully."""
    input_features = [
        "initial_batter",
        "initial_bowler",
        "num_batsmen",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
    ]
    target = "runs"

    X = training_data[input_features]
    y = training_data[target]

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    assert hasattr(model, "predict"), "Model training failed, no predict method found"
    assert isinstance(
        model, RandomForestRegressor
    ), "Model is not an instance of RandomForestRegressor"


def test_model_saving(training_data):
    """Test if the model saves correctly."""
    input_features = [
        "initial_batter",
        "initial_bowler",
        "num_batsmen",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
    ]
    target = "runs"

    X = training_data[input_features]
    y = training_data[target]

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    joblib.dump(model, model_file)
    assert model_file.exists(), "Model file was not saved"
    loaded_model = joblib.load(model_file)
    assert isinstance(
        loaded_model, RandomForestRegressor
    ), "Loaded model is not the correct type"


def test_training_metrics_logging(caplog):
    """Test if training logs metrics correctly."""
    with patch("train.data_folder", data_folder):
        with caplog.at_level(logging.INFO):
            main()

    assert any(
        "Training MAE" in record.message for record in caplog.records
    ), "Training MAE not logged"
    assert any(
        "Test MAE" in record.message for record in caplog.records
    ), "Test MAE not logged"
    assert any(
        "Training RMSE" in record.message for record in caplog.records
    ), "Training RMSE not logged"
    assert any(
        "Test RMSE" in record.message for record in caplog.records
    ), "Test RMSE not logged"


if __name__ == "__main__":
    results = {"status": "success", "errors": []}
    try:
        pytest.main([__file__])
    except SystemExit as e:
        if e.code != 0:
            results["status"] = "failure"
            results["errors"].append(f"Test suite failed with exit code {e.code}")
            with open(log_file, "w") as f:
                json.dump(results, f)
            exit(1)

    with open(log_file, "w") as f:
        json.dump(results, f)
