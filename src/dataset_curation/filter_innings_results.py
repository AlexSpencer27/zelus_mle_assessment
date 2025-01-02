import pandas as pd
from collections import defaultdict
from pathlib import Path
import os
import tqdm
import warnings

warnings.filterwarnings("ignore")

# data folder is one up from this script
script_folder = Path(__file__).parent
data_folder = script_folder.parent.parent / "data"
output_folder = data_folder / "intermediate"
os.makedirs(output_folder, exist_ok=True)


def main():
    # read the innings results
    print("Reading innings results")
    innings_results = pd.read_parquet(
        os.path.join(data_folder, "parsed", "innings_results.parquet")
    )

    # read the match results
    print("Reading match results")
    match_results = pd.read_parquet(
        os.path.join(data_folder, "parsed", "match_results.parquet")
    )

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

    output_dict = defaultdict(list)

    # count duplicate rows
    duplicate_rows = innings_results.duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows} / {len(innings_results)}")

    innings_results = innings_results.drop_duplicates()

    # replace wicket.kind of "retired hurt" with None
    wickets_no_loss = ["retired hurt"]  # , "obstructing the field"]
    for wicket_kind in wickets_no_loss:
        innings_results.loc[innings_results["wicket.kind"] == wicket_kind] = None

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

    print("Converting to DataFrame")
    output_df = pd.DataFrame(output_dict)

    print("Converting to parquet")
    output_df.to_parquet(
        os.path.join(output_folder, "filtered_innings.parquet"),
        index=False,
    )
    print("Done")


def filter_non_results(innings_results, match_results):
    null_matchid = set(match_results[match_results["result"] == "no result"]["matchid"])
    innings_results = innings_results[~innings_results["matchid"].isin(null_matchid)]

    return innings_results


def filter_male_matches(innings_results, match_results):
    nonmale_matchid = set(match_results[match_results["gender"] != "male"]["matchid"])
    innings_results = innings_results[~innings_results["matchid"].isin(nonmale_matchid)]

    return innings_results


def get_match_metadata(match_results, match_id, team):
    match_meta = dict()
    # get the match stats for that match_id
    match_result = match_results[match_results["matchid"] == match_id]

    # ensure the match_winner is the current group's team
    match_meta["date"] = match_result["dates"].values[0]

    match_meta["winner"] = match_result["outcome.winner"].values[0]

    match_meta["overs"] = match_result["overs"].values[0]

    match_meta["wickets"] = match_result["outcome.wickets"].values[0]

    match_meta["runs"] = match_result["outcome.runs"].values[0]

    match_meta["method"] = match_result["outcome.method"].values[0]

    match_meta["opponent"] = match_result[match_result["teams"] != team][
        "teams"
    ].values[0]

    return match_meta


def get_remaining_overs(df, match_meta):
    df["over_int"] = df["over"].apply(lambda x: int(float(x)) + 1)
    df["remaining_overs"] = match_meta["overs"] - df["over_int"]

    return df


def get_remaining_wickets(df):
    # convert wicket.kind to a binary feature, 0 if wicket.kind is not null, 1 otherwise
    df["wicket.binary"] = df["wicket.kind"].apply(lambda x: 0 if x is None else 1)

    # get the number of wickets taken in the innings
    df["wickets"] = df["wicket.binary"].cumsum()

    # get the remaining wickets in the innings
    df["remaining_wickets"] = 10 - df["wickets"]

    # drop the extraneous features
    for key in ["wicket.binary", "wickets"]:
        df = df.drop(columns=key)

    return df


def encode_by_order(df, key):
    unique_batters = df[key].unique()
    batter_number_dict = {batter: i + 1 for i, batter in enumerate(unique_batters)}
    df[f"{key}_number"] = df[key].apply(lambda x: batter_number_dict[x])
    return df


if __name__ == "__main__":
    main()
