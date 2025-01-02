import pandas as pd
from pathlib import Path
import os
import pytest
import json

# Define data path
data_folder = Path(__file__).parents[2] / "data"
training_data_path = data_folder / "training/training_data.parquet"

test_result_folder = data_folder / "tests"
os.makedirs(test_result_folder, exist_ok=True)
log_file = test_result_folder / "test_training_data.json"

# Load training data
training_df = pd.read_parquet(training_data_path)


def test_no_missing_values() -> None:
    """
    Ensure there are no missing values in critical columns.
    """
    critical_columns = [
        "initial_batter",
        "initial_bowler",
        "num_batsmen",
        "num_bowlers",
        "num_deliveries",
        "remaining_wickets",
        "remaining_overs",
        "runs",
    ]
    missing_counts = training_df[critical_columns].isnull().sum()
    assert missing_counts.sum() == 0, f"Missing values found: {missing_counts}"


def test_value_ranges() -> None:
    """
    Check that numerical columns fall within expected ranges.
    """
    for key in ["num_batsmen", "num_bowlers", "initial_batter", "initial_bowler"]:
        assert training_df[key].between(1, 11).all(), f"Invalid values in '{key}'"

    assert (
        training_df["remaining_wickets"].between(0, 10).all()
    ), "Invalid values in 'remaining_wickets'"

    assert (
        training_df["remaining_overs"].between(0, 50).all()
    ), "Invalid values in 'remaining_overs'"

    assert training_df["runs"].between(0, 100000).all(), "Invalid values in 'runs'"


def test_no_duplicate_rows() -> None:
    """
    Ensure there are no duplicate rows in the training data.
    """
    duplicates = training_df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows in the training data"


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
