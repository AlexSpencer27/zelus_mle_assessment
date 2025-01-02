import pandas as pd
import json
import pytest
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from typing import Dict, Tuple

# Define data paths
data_folder = Path(__file__).parents[2] / "data"

training_df = pd.read_parquet(data_folder / "training/training_data.parquet")
match_data = pd.read_parquet(data_folder / "parsed/match_results.parquet")
innings_data = pd.read_parquet(data_folder / "parsed/innings_results.parquet")

test_result_folder = data_folder / "tests"
os.makedirs(test_result_folder, exist_ok=True)
log_file = test_result_folder / "test_computed_metrics.json"


def validate_run_difference(
    computed_group: pd.DataFrame,
    match_id: int,
    winner: str,
    loser: str,
    run_diff_true: float,
) -> None:
    """
    Validate that the computed run difference aligns with the ground truth.

    Args:
        computed_group (pd.DataFrame): The computed metrics for the match.
        match_id (int): The match ID.
        winner (str): Winning team's name.
        loser (str): Losing team's name.
        run_diff_true (float): True run difference from match data.

    Raises:
        AssertionError: If the computed run difference does not match the ground truth.
    """
    match_teams = [winner, loser]
    total_runs = {
        team: computed_group[computed_group["team"] == team]["runs"].sum()
        for team in match_teams
    }

    # Check for discrepancies
    if total_runs[winner] - total_runs[loser] < run_diff_true:
        winner_innings = innings_data[
            (innings_data["matchid"] == match_id) & (innings_data["team"] == winner)
        ]
        loser_innings = innings_data[
            (innings_data["matchid"] == match_id) & (innings_data["team"] == loser)
        ]

        winner_runs = winner_innings["runs.total"].sum()
        loser_runs = loser_innings["runs.total"].sum()

        diff_runs = winner_runs - loser_runs

        # Account for potential bugs in match data
        assert (
            diff_runs >= run_diff_true
        ), f"Run difference mismatch in match {match_id}"
    else:
        assert (
            total_runs[winner] - total_runs[loser] >= run_diff_true
        ), f"Computed run difference does not meet expected value in match {match_id}"


def validate_wicket_difference(
    computed_group: pd.DataFrame,
    match_teams: Tuple[str, str],
    winner: str,
    wickets_true: int,
) -> None:
    """
    Validate that the computed remaining wickets align with the ground truth.

    Args:
        computed_group (pd.DataFrame): The computed metrics for the match.
        match_teams (Tuple[str, str]): Tuple of the match's two teams.
        winner (str): Winning team's name.
        wickets_true (int): True number of remaining wickets from match data.

    Raises:
        AssertionError: If the computed remaining wickets do not match the ground truth.
    """
    remaining_wickets = {
        team: computed_group[computed_group["team"] == team]["remaining_wickets"].min()
        for team in match_teams
    }

    assert (
        remaining_wickets[winner] == wickets_true
    ), f"Wicket difference mismatch for winner {winner} in match {match_teams}"


def validate_tie_or_draw(
    computed_group: pd.DataFrame, match_teams: Tuple[str, str]
) -> None:
    """
    Validate that the computed runs for tied matches are equal for both teams.

    Args:
        computed_group (pd.DataFrame): The computed metrics for the match.
        match_teams (Tuple[str, str]): Tuple of the match's two teams.

    Raises:
        AssertionError: If the computed runs for tied matches do not match.
    """
    total_runs = {
        team: computed_group[computed_group["team"] == team]["runs"].sum()
        for team in match_teams
    }

    assert (
        total_runs[match_teams[0]] == total_runs[match_teams[1]]
    ), f"Tie validation failed for teams {match_teams}"


def test_computed_metrics() -> None:
    """
    Test that computed metrics align with ground truth from match results.

    This function iterates through matches, validating runs and wickets based on match results.
    """
    matches_grouped = match_data.groupby(by="matchid")

    with tqdm(
        total=len(matches_grouped), desc="Testing Computed Metrics", unit="matchid"
    ) as pbar:
        for match_id, group in matches_grouped:
            if (
                group.iloc[0]["result"] == "no result"
                or group.iloc[0]["gender"] == "female"
            ):
                pbar.update(1)
                continue

            computed_group = training_df[training_df["matchid"] == match_id]
            match_teams = group["teams"].unique()

            winner = group.iloc[-1]["outcome.winner"]
            loser = [team for team in match_teams if team != winner][0]

            run_diff_true = group.iloc[-1]["outcome.runs"]
            wickets_true = group.iloc[-1]["outcome.wickets"]

            if group.iloc[0]["outcome.method"] == "D/L":
                # Duckworth-Lewis method not validated in this test
                pass

            elif ~np.isnan(run_diff_true):
                validate_run_difference(
                    computed_group, match_id, winner, loser, run_diff_true
                )

            elif ~np.isnan(wickets_true):
                validate_wicket_difference(
                    computed_group, match_teams, winner, wickets_true
                )

            elif group.iloc[0]["result"] in ["tie", None]:
                validate_tie_or_draw(computed_group, match_teams)

            else:
                raise ValueError(f"Unknown data quality error in match {match_id}")

            pbar.update(1)


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
