import pandas as pd
from collections import defaultdict
from pathlib import Path
import os
import tqdm
from typing import Dict, List, Any

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")

# Define paths
script_folder: Path = Path(__file__).parent
data_folder: Path = script_folder.parent.parent / "data"
output_folder: Path = data_folder / "intermediate"
os.makedirs(output_folder, exist_ok=True)


def main() -> None:
    """
    Main function to process and filter innings results, enrich with metadata,
    and save the resulting dataset as a parquet file.
    """
    # Read the innings and match results
    print("Reading innings results")
    innings_results: pd.DataFrame = pd.read_parquet(
        os.path.join(data_folder, "parsed", "innings_results.parquet")
    )

    print("Reading match results")
    match_results: pd.DataFrame = pd.read_parquet(
        os.path.join(data_folder, "parsed", "match_results.parquet")
    )

    # Filter out non-results and non-male matches
    innings_results = filter_non_results(innings_results, match_results)
    innings_results = filter_male_matches(innings_results, match_results)

    key_columns = [
        "batsman",
        "batsman_number",
        "bowler",
        "bowler_number",
        "over",
        "over_int",
        "remaining_overs",
        "team",
        "opponent",
        "innings",
        "matchid",
        "date",
        "wicket.kind",
        "remaining_wickets",
        "runs.batsman",
        "runs.extras",
        "runs.total",
    ]

    output_dict: Dict[str, List[Any]] = defaultdict(list)

    # Count and remove duplicate rows
    duplicate_rows = innings_results.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows} / {len(innings_results)}")
    innings_results = innings_results.drop_duplicates()

    # Replace certain wicket kinds with None
    wickets_no_loss = ["retired hurt"]
    for wicket_kind in wickets_no_loss:
        innings_results.loc[
            innings_results["wicket.kind"] == wicket_kind, "wicket.kind"
        ] = None

    # Group data by match and innings
    innings_grouped = innings_results.groupby(by=["matchid", "innings"])
    pbar = tqdm.tqdm(total=len(innings_grouped), desc="Parsing innings results")

    for (match_id, inning), group in innings_grouped:
        match_meta = get_match_metadata(
            match_results, match_id, group["team"].values[0]
        )

        group = get_remaining_overs(group, match_meta)
        group = get_remaining_wickets(group)

        for key in ["batsman", "bowler"]:
            group = encode_by_order(group, key)

        group["opponent"] = match_meta["opponent"]
        group["date"] = match_meta["date"]

        for key in key_columns:
            output_dict[key].extend(group[key].tolist())

        pbar.update(1)
    pbar.close()

    # Convert to DataFrame and save
    print("Converting to DataFrame")
    output_df: pd.DataFrame = pd.DataFrame(output_dict)

    print("Saving to parquet")
    output_df.to_parquet(
        os.path.join(output_folder, "filtered_innings.parquet"),
        index=False,
    )
    print("Done")


def filter_non_results(
    innings_results: pd.DataFrame, match_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter out matches with no result.
    """
    null_matchid = set(match_results[match_results["result"] == "no result"]["matchid"])
    return innings_results[~innings_results["matchid"].isin(null_matchid)]


def filter_male_matches(
    innings_results: pd.DataFrame, match_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter out matches that are not categorized as male.
    """
    nonmale_matchid = set(match_results[match_results["gender"] != "male"]["matchid"])
    return innings_results[~innings_results["matchid"].isin(nonmale_matchid)]


def get_match_metadata(
    match_results: pd.DataFrame, match_id: int, team: str
) -> Dict[str, Any]:
    """
    Extract metadata for a given match.
    """
    match_result = match_results[match_results["matchid"] == match_id]
    return {
        "date": match_result["dates"].values[0],
        "winner": match_result["outcome.winner"].values[0],
        "overs": match_result["overs"].values[0],
        "wickets": match_result["outcome.wickets"].values[0],
        "runs": match_result["outcome.runs"].values[0],
        "method": match_result["outcome.method"].values[0],
        "opponent": match_result[match_result["teams"] != team]["teams"].values[0],
    }


def get_remaining_overs(df: pd.DataFrame, match_meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute remaining overs for each delivery.
    """
    df["over_int"] = df["over"].apply(lambda x: int(float(x)) + 1)
    df["remaining_overs"] = match_meta["overs"] - df["over_int"]
    return df


def get_remaining_wickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute remaining wickets for each delivery.
    """
    df["wicket.binary"] = df["wicket.kind"].apply(lambda x: 0 if x is None else 1)
    df["wickets"] = df["wicket.binary"].cumsum()
    df["remaining_wickets"] = 10 - df["wickets"]
    return df.drop(columns=["wicket.binary", "wickets"])


def encode_by_order(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Encode a column by its order of appearance.
    """
    unique_values = df[key].unique()
    value_mapping = {value: i + 1 for i, value in enumerate(unique_values)}
    df[f"{key}_number"] = df[key].map(value_mapping)
    return df


if __name__ == "__main__":
    main()
