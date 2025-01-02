import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import os
import pytest
import json

# Define data paths
data_folder = Path(__file__).parents[2] / "data"

filtered_innings_df = pd.read_parquet(
    data_folder / "intermediate/filtered_innings.parquet"
)
match_data = pd.read_parquet(data_folder / "parsed/match_results.parquet")


test_result_folder = data_folder / "tests"
os.makedirs(test_result_folder, exist_ok=True)
log_file = test_result_folder / "test_innings_endings.json"


def get_match_metadata(
    match_results: pd.DataFrame, match_id: int, team: str
) -> Dict[str, any]:
    """
    Fetch metadata for a given match and team.

    Args:
        match_results (pd.DataFrame): DataFrame containing match results.
        match_id (int): Match identifier.
        team (str): Team name for the current inning.

    Returns:
        Dict[str, any]: Metadata dictionary for the match.
    """
    match_result = match_results[match_results["matchid"] == match_id]

    return {
        "winner": match_result["outcome.winner"].iloc[0],
        "overs": match_result["overs"].iloc[0],
        "wickets": match_result["outcome.wickets"].iloc[0],
        "runs": match_result["outcome.runs"].iloc[0],
        "method": match_result["outcome.method"].iloc[0],
        "opponent": match_result[match_result["teams"] != team]["teams"].iloc[0],
    }


def test_inning_endings() -> None:
    """
    Test and categorize how innings end to ensure data integrity.

    Raises:
        AssertionError: If any inning ends in an unknown or invalid manner.
    """
    inning_ends: Dict[str, int] = defaultdict(int)
    innings_grouped = filtered_innings_df.groupby(by=["matchid", "innings"])

    # Initialize the progress bar
    with tqdm(
        total=len(innings_grouped), desc="Testing Inning Endings", unit="inning"
    ) as pbar:
        for (match_id, inning), group in innings_grouped:
            # Fetch match metadata for the current group
            team = group["team"].iloc[0]
            match_meta = get_match_metadata(match_data, match_id, team)

            last_remaining_wickets = group.iloc[-1]["remaining_wickets"]
            last_remaining_overs = group.iloc[-1]["remaining_overs"]

            if last_remaining_overs != 0 and last_remaining_wickets != 0:
                # Handle special ending cases
                if match_meta["method"] == "D/L":
                    inning_ends["Duckworth Lewis"] += 1
                elif not np.isnan(match_meta["wickets"]):
                    inning_ends["wickets"] += 1
                elif match_meta["winner"] == team and inning == 2:
                    inning_ends["winner_in_2nd"] += 1
                elif not np.isnan(match_meta["runs"]):
                    process_run_based_ending(group, match_id, match_meta, inning_ends)
                else:
                    raise AssertionError(
                        f"Match ID: {match_id} - Unknown inning ending"
                    )
            else:
                # Handle standard endings
                if last_remaining_overs == 0:
                    inning_ends["out_of_overs"] += 1
                elif last_remaining_wickets == 0:
                    inning_ends["out_of_wickets"] += 1

            # Update the progress bar
            pbar.update(1)

    # Print breakdown of inning endings
    print_inning_ends(inning_ends)


def process_run_based_ending(
    group: pd.DataFrame,
    match_id: int,
    match_meta: Dict[str, any],
    inning_ends: Dict[str, int],
) -> None:
    """
    Handle innings that end based on runs.

    Args:
        group (pd.DataFrame): Data for the current inning.
        match_id (int): Match identifier.
        match_meta (Dict[str, any]): Metadata for the match.
        inning_ends (Dict[str, int]): Counter for inning ending types.
    """
    team_runs = group["runs.total"].sum()

    # Fetch opponent's innings data
    other_team_innings = filtered_innings_df[
        (filtered_innings_df["matchid"] == match_id)
        & (filtered_innings_df["team"] == match_meta["opponent"])
    ]
    other_team_total_runs = other_team_innings["runs.total"].sum()

    # Determine if the ending is valid
    assert (
        np.abs(team_runs - other_team_total_runs) > 0
    ), f"Match ID: {match_id} - Invalid loser logic"
    inning_ends["loser_by_runs"] += 1


def print_inning_ends(inning_ends: Dict[str, int]) -> None:
    """
    Print inning ending statistics.

    Args:
        inning_ends (Dict[str, int]): Counter for inning ending types.
    """
    total_results = sum(inning_ends.values())
    print("Inning Endings Breakdown:")
    for key, value in inning_ends.items():
        percentage = (value / total_results) * 100
        print(f"{key}: {percentage:.2f}%")


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
