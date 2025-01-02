import pandas as pd
from pathlib import Path
import os

# data folder is one up from this script
script_folder = Path(__file__).parent
data_folder = script_folder.parent.parent / "data"

output_folder = data_folder / "intermediate"
os.makedirs(output_folder, exist_ok=True)


def main():
    filtered_innings_file = os.path.join(
        data_folder, "intermediate", "filtered_innings.parquet"
    )

    print("Reading filtered innings results")
    df = pd.read_parquet(filtered_innings_file)

    # create smaller df for target output
    key_cols = dict(
        matchid=dict(dtp=int, rename="match_id"),
        date=dict(dtp=str, rename="date"),
        team=dict(dtp=str, rename="batting_team"),
        opponent=dict(dtp=str, rename="bowling_team"),
        innings=dict(dtp=int, rename="innings_order"),
        remaining_overs=dict(dtp=int, rename="remaining_overs"),
        remaining_wickets=dict(dtp=int, rename="remaining_wickets"),
    )

    print("Taking subset of columns")
    df_out = pd.DataFrame()
    for key, val in key_cols.items():
        df_out[val["rename"]] = df[key].astype(val["dtp"])

    print("Writing to csv")
    df_out.to_csv(os.path.join(output_folder, "q3a.csv"), index=False)
    print("Done")


if __name__ == "__main__":
    main()
