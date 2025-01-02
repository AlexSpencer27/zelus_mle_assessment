import pytest
from typer.testing import CliRunner
from pathlib import Path
import pandas as pd
import sys
import os
import json

# Import the main app
script_folder = Path(__file__).parents[2] / "src" / "model_package"
sys.path.append(str(script_folder))
from run_model import app

data_folder = Path(__file__).parents[2] / "data"

test_result_folder = data_folder / "tests"
os.makedirs(test_result_folder, exist_ok=True)
log_file = test_result_folder / "test_model_interaction.json"

# Initialize Typer test runner
runner = CliRunner()

# Mock data for testing
MOCK_DATA_PATH = Path(__file__).parent / "mock_data.parquet"


@pytest.fixture
def mock_data():
    """Create a mock dataset for testing."""
    data = pd.DataFrame(
        {
            "matchid": [1, 2, 3, 4],
            "team": ["India", "Australia", "England", "Pakistan"],
            "opponent": ["England", "India", "Pakistan", "Australia"],
            "over_num": [1, 5, 10, 20],
            "initial_batter": [1, 2, 3, 4],
            "initial_bowler": [1, 1, 2, 2],
            "num_batsmen": [2, 3, 4, 2],
            "num_bowlers": [1, 1, 2, 1],
            "num_deliveries": [6, 6, 6, 6],
            "remaining_wickets": [10, 9, 8, 7],
            "remaining_overs": [50, 45, 40, 35],
            "date": ["2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04"],
        }
    )
    data.to_parquet(MOCK_DATA_PATH)
    yield data
    MOCK_DATA_PATH.unlink()


def test_valid_inputs(mock_data):
    """Test the script with valid inputs."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
            "--start-over",
            "1",
            "--end-over",
            "5",
            "--num-matches",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "predicted_runs" in result.stdout


def test_mixed_case(mock_data):
    """Test the script with valid inputs."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "InDiA",
            "--bowling-team",
            "englaNd",
            "--start-over",
            "1",
            "--end-over",
            "5",
            "--num-matches",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "predicted_runs" in result.stdout


def test_invalid_model_path(mock_data):
    """Test the script with an invalid model path."""
    result = runner.invoke(
        app,
        [
            "--model",
            "invalid_model_path.pkl",  # Invalid path
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid model path" in result.output


def test_invalid_team(mock_data):
    """Test the script with an invalid team."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "InvalidTeam",
            "--bowling-team",
            "England",
        ],
    )
    assert result.exit_code == 2
    assert "Batting team 'InvalidTeam' not found" in result.stdout


def test_invalid_opponent(mock_data):
    """Test the script with an invalid opponent."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "InvalidOpponent",
        ],
    )
    assert result.exit_code == 2
    assert "Bowling team 'InvalidOpponent' never played India." in result.stdout


def test_invalid_over_range(mock_data):
    """Test the script with an invalid over range."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
            "--start-over",
            "10",
            "--end-over",
            "5",  # Invalid range
        ],
    )
    assert result.exit_code == 2
    assert "Start over must be less than or equal to end over" in result.stdout


def test_edge_case_first_over(mock_data):
    """Test predictions for the first over only."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
            "--start-over",
            "1",
            "--end-over",
            "1",  # Only first over
            "--num-matches",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "predicted_runs" in result.stdout


def test_all_bowling_teams(mock_data):
    """Test predictions with no bowling team specified."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "None",
            "--start-over",
            "1",
            "--end-over",
            "5",
            "--num-matches",
            "10",
        ],
    )
    assert result.exit_code == 0
    assert "predicted_runs" in result.stdout


def test_edge_case_all_overs(mock_data):
    """Test predictions for all overs."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
            "--start-over",
            "1",
            "--end-over",
            "50",  # All overs
            "--num-matches",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "predicted_runs" in result.stdout


def test_edge_case_all_games(mock_data):
    """Test predictions for all overs."""
    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
            "--start-over",
            "1",
            "--end-over",
            "50",  # All overs
            "--num-matches",
            "-1",
        ],
    )
    assert result.exit_code == 0
    assert "predicted_runs" in result.stdout


def test_missing_columns(mock_data):
    """Test the script with missing critical columns."""
    incomplete_data = mock_data.drop(columns=["initial_batter"])
    incomplete_data.to_parquet(MOCK_DATA_PATH)

    result = runner.invoke(
        app,
        [
            "--data",
            str(MOCK_DATA_PATH),
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
        ],
    )
    assert result.exit_code == 2
    assert "Dataset is missing required columns: initial_batter" in result.stdout


def test_invalid_data_path():
    """Test the script with an invalid data file path."""
    result = runner.invoke(
        app,
        [
            "--data",
            "invalid_path.parquet",  # Invalid path
            "--batting-team",
            "India",
            "--bowling-team",
            "England",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid data file path" in result.stdout


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
